"""Tile the wide radial gate projection together with gather/TP/scatter.

Frozen-weight force inference saves the narrow radial hidden state instead of
the complete edge/path/channel gate. First coordinate adjoints rebuild gates
per tile and immediately contract their cotangents back to the narrow state.
Higher derivatives use the ordinary differentiable formula. Trainable final
gate weights can opt into paged first/mixed adjoints. Trainable TP weights keep
the existing general message route.
"""
import torch
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch._subclasses.fake_tensor import is_fake

from .adjoint_requirements import required_gradients
from .bond_projection import _forward_raw as project, _transpose_raw as transpose
from .monomial_operator import triton
from .rematerialized_message import _edge_slice, _tiles, _native


def _gate_projection(h,a,bias,p):
    if p.get('gate_gemm_policy','torch')=='dtk2604' and not bias.numel():
        from .hip_gate_gemm import gemm
        out=h.new_empty((h.shape[0],a.shape[0]))
        if gemm(h,a,out,mode='forward'):return out
        del out
    return F.linear(h,a,bias if bias.numel() else None)


def _gate_adjoint(dg,a,dh,p):
    """Bound row tiles independently of the TP chunk without extra workspace."""
    rows=int(p.get('gate_adjoint_rows',0))
    matrix=dg.reshape(dg.shape[0],-1)
    if matrix.shape[0]==0:return
    rows=rows or matrix.shape[0]
    for start in range(0,matrix.shape[0],rows):
        block=matrix[start:start+rows];out=dh[start:start+rows]
        if p.get('gate_gemm_policy','torch')=='dtk2604':
            from .hip_gate_gemm import gemm
            if gemm(block,a,out,mode='grad'):continue
        torch.mm(block,a,out=out)


def _adjoint_tiles(edges,p,cached):
    """Slice cached forward gates while bounding the transient TP cotangent."""
    rows=int(p.get('gate_adjoint_edges',0))
    for begin,end in _tiles(edges,p):
        gate=cached.get(begin)
        if rows<=0 or end-begin<=rows:
            yield begin,end,gate
        else:
            for start in range(begin,end,rows):
                stop=min(start+rows,end)
                yield start,stop,None if gate is None else gate[start-begin:stop-begin]


class _StreamedGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,sh,w,h,a,bias,src,dst,mask,p,projection):
        from .fused_message import _group_message_raw,_normal_message_output,_channel_node
        ctx.plan=p;ctx.projection=projection
        cached={};remaining=p.get("gate_cache_bytes",0)
        canonical=p.get("generated_layout","packed")=="canonical" and p.get("generated_forward",False)
        native_packed=bool(p.get('native_packed',False))
        nout=p.get('output_nodes',x.shape[0])
        out=x.new_zeros(nout,p["odim"]);xc=x if canonical or native_packed else _channel_node(x,p)
        if src.numel() and p.get('generated_csr_fused_adjoint',False) and any(v.requires_grad for v in (x,sh,h)):
            from .generated_csr_fused_adjoint import _extension
            _extension(p,projection)
        segmented=p.get("segmented_forward",False)
        if segmented:
            from .segmented_message import destination_plan,segmented_forward
            ptr,host_ptr=destination_plan(dst,x.shape[0])
        csr=None
        from .generated_csr_forward import enabled
        if enabled(p.get("generated_csr_forward",False),src.numel(),nout):
            from .generated_csr_forward import topology
            csr=topology(dst,nout,p.get('generated_topology','native'),p.get('csr_slice_search',False))
        for start,end in _tiles(src.numel(),p):
            g=_gate_projection(h[start:end],a,bias,p).reshape(end-start,p["p"],p["c"])
            bytes_used=g.numel()*g.element_size()
            if bytes_used<=remaining:
                cached[start]=g;remaining-=bytes_used
            if csr is not None:
                from .generated_csr_forward import forward
                forward(xc,sh[start:end],w,g,src[start:end],_edge_slice(mask,start,end),out,p,projection,csr,start)
                del g
                continue
            if p.get("generated_forward",False):
                from .generated_cartesian_message import generated_forward
                generated_forward(xc,sh[start:end],w,g,src[start:end],dst[start:end],_edge_slice(mask,start,end),out,p,projection)
                del g
                continue
            if p.get("cached_edge_forward",False):
                from .cached_edge_message import cached_forward
                cached_forward(xc,sh[start:end],w,g,src[start:end],dst[start:end],_edge_slice(mask,start,end),out,p,projection)
                del g
                continue
            magnetic=project(sh[start:end],projection)
            if segmented:
                segmented_forward(xc,magnetic,w,g,src[start:end],_edge_slice(mask,start,end),out,p,ptr,host_ptr,start,end)
            else:
                _group_message_raw(x,magnetic,w,g,src[start:end],dst[start:end],
                                   _edge_slice(mask,start,end),p,output=out,channel_node=xc)
            del g,magnetic
        ctx.cached_gate_starts=tuple(cached)
        ctx.save_for_backward(x,sh,w,h,a,bias,src,dst,mask,*cached.values())
        return out if canonical or native_packed or p.get('inference_degree_layout',False) else _normal_message_output(out,p)

    @staticmethod
    def backward(ctx,grad):
        from .fused_message import _vjp_raw,_normal_feature_gradient,_channel_inputs
        x,sh,w,h,a,bias,src,dst,mask,*cache=ctx.saved_tensors;p=ctx.plan;projection=ctx.projection
        cached=dict(zip(ctx.cached_gate_starts,cache))
        needed=required_gradients((x,sh,w,h,a,bias),ctx.needs_input_grad[:6])
        grad=grad.contiguous()
        if torch.is_grad_enabled():
            local=tuple(value.clone() for value in (x,sh,w,h,a,bias))
            lx,lsh,lw,lh,la,lb=local
            gate=F.linear(lh,la,lb if lb.numel() else None).reshape(src.numel(),p["p"],p["c"])
            output=_native(lx,lsh,lw,gate,src,dst,mask,p,projection)
            if 'output_nodes' in p:
                output=output[:p['output_nodes']]
            if p.get('inference_degree_layout',False):
                parts=[];n=output.shape[0]
                for degree in range(p['lmax']+1):
                    width=p['c']*p['counts'].get(degree,0)*(2*degree+1);start=p['ooffset'][degree]
                    parts.append(output[:,start:start+width].reshape(n,-1,2*degree+1).transpose(-2,-1).reshape(-1))
                output=torch.cat(parts).reshape(n,p['odim'])
            active=[value for value,need in zip(local,needed) if need]
            values=iter(torch.autograd.grad(output,active,grad,create_graph=True,allow_unused=True))
            grads=[next(values) if need else None for need in needed]
            return *grads,None,None,None,None,None
        # The entry guard rejects live final gate and TP weights. Do not silently
        # return missing parameter gradients if that guard changes in future.
        if any(needed[i] for i in (2,4,5)):
            raise RuntimeError("Streamed frozen-gate inference received a trainable weight")
        dx=torch.zeros_like(x) if needed[0] else None
        dsh=torch.empty_like(sh) if needed[1] else None
        dh=torch.empty_like(h) if needed[3] else None
        canonical=p.get("generated_layout","packed")=="canonical" and p.get("generated_adjoint",False)
        native_packed=bool(p.get('native_packed',False))
        packed=None
        if src.numel():
            if p.get('inference_degree_layout',False):
                from .fused_message import _channel_node
                packed=(_channel_node(x,p),grad)
            else:packed=(x,grad) if canonical or native_packed else _channel_inputs(x,grad,p)
        fused_source=bool(p.get('generated_csr_fused_adjoint',False) and p.get('generated_adjoint',False))
        if fused_source:
            from .generated_csr_forward import topology
            from .generated_csr_fused_adjoint import adjoint
            source_csr=topology(src,x.shape[0],p.get('generated_topology','native'))
            if dsh is not None:dsh.zero_()
        from .inference_adjoint_schedule import adjoint_plan
        generated_plan=(adjoint_plan(p,x,src.numel(),needed[0],need_geometry=needed[1],need_gate=needed[3])
                        if p.get('generated_adjoint',False) else p)
        for start,end,g in _adjoint_tiles(src.numel(),p,cached):
            if g is None:g=_gate_projection(h[start:end],a,bias,p).reshape(end-start,p["p"],p["c"])
            if fused_source:
                dg=torch.empty_like(g) if needed[3] else None
                adjoint(packed[0],sh[start:end],w,g,dst[start:end],_edge_slice(mask,start,end),packed[1],
                    dx,dsh[start:end] if dsh is not None else None,dg,p,projection,source_csr,start,
                    (needed[0],needed[1],needed[3]))
                if dg is not None:_gate_adjoint(dg,a,dh[start:end],p)
                del g,dg
                continue
            if p.get("generated_adjoint",False):
                from .generated_cartesian_message import generated_adjoint
                # Tile only the joint feature/geometry/gate adjoint. First-layer
                # geometry-only work keeps its existing, faster schedule. The
                # tiled kernel reduces into the same dsh/dg destinations.
                ds,dg=generated_adjoint(packed[0],sh[start:end],w,g,src[start:end],dst[start:end],
                    _edge_slice(mask,start,end),packed[1],dx,generated_plan,projection,(needed[0],needed[1],needed[3]),
                    harmonic_output=dsh[start:end] if dsh is not None else None)
                if dg is not None:_gate_adjoint(dg,a,dh[start:end],p)
                del g,ds,dg
                continue
            magnetic=project(sh[start:end],projection)
            _,db,_,dg=_vjp_raw(x,magnetic,w,g,src[start:end],dst[start:end],
                _edge_slice(mask,start,end),grad,p,(needed[0],needed[1],False,needed[3]),
                feature_output=dx,channel_inputs=packed,harmonic_output=dsh[start:end] if dsh is not None else None,
                projection=projection)
            if db is not None:dsh[start:end].copy_(transpose(db,projection))
            if dg is not None:_gate_adjoint(dg,a,dh[start:end],p)
            del g,magnetic,db,dg
        if dx is not None and not (canonical or native_packed):dx=_normal_feature_gradient(dx,p)
        return dx,dsh,None,dh,None,None,None,None,None,None,None


def streamed_gate_message(tp,x1,edge_attrs,edge_feats,fc,src,dst,mask,*,output_nodes=None):
    """Return path-preserving node blocks, or None for an unsupported context."""
    sample=next(iter(x1.values()))
    if (torch.compiler.is_compiling() or torch._C._are_functorch_transforms_active()
            or get_proxy_mode() is not None or is_fake(sample)):
        return None
    e=src.numel()
    if not isinstance(fc,torch.nn.Sequential) or not len(fc):return None
    last=fc[-1]
    train_gate=getattr(tp,'_training_gate_schedule','materialized')!='materialized'
    if (triton is None or not isinstance(last,torch.nn.Linear)
            or (not train_gate and (last.weight.requires_grad or (last.bias is not None and last.bias.requires_grad)))
            or tp.weight.requires_grad
            or not sample.is_cuda or sample.dtype!=torch.float32 or tp.internal_compute_dtype!=torch.float32
            or edge_feats.dtype!=sample.dtype or edge_feats.device!=sample.device or edge_feats.ndim!=2
            or edge_feats.shape[0]!=e or last.out_features!=tp.num_paths*tp.channels
            or tp.channels<1 or tp.num_paths==0
            or any(v.ndim!=3 or v.dtype!=sample.dtype or v.device!=sample.device for v in x1.values())
            or any(v.ndim!=3 or v.shape[-2]!=1 or v.dtype!=sample.dtype or v.device!=sample.device for v in edge_attrs.values())
            or torch.compiler.is_compiling() or torch.is_autocast_enabled(sample.device.type)
            or torch._C._are_functorch_transforms_active() or get_proxy_mode() is not None or is_fake(sample)
            or (mask is not None and mask.requires_grad)
            or any(int(group["l1"]) not in x1 or int(group["l2"]) not in edge_attrs for group in tp._groups)):
        return None
    from .fused_message import _plan
    from .bond_projection import projection_plan
    projectors=tp._get_bond_projectors(sample.device,torch.float32)
    if any(value.requires_grad for value in projectors):return None
    projection=projection_plan(tp,projectors,sample.device,sample.dtype)
    plan={**_plan(tp,sorted(x1),sample.device,True),"edge_chunk_size":getattr(tp,"_message_chunk_edges",65536),
          "fused_path_forward":getattr(tp,"_fused_path_forward",False),
          "fused_path_adjoint":getattr(tp,"_fused_path_adjoint",False),
          "path_tile_edges":getattr(tp,"_path_tile_edges",1),"path_warps":getattr(tp,"_path_warps",4),
          "geometry_by_column":getattr(tp,"_geometry_by_column",False),"geometry_tile_edges":getattr(tp,"_geometry_tile_edges",32)}
    plan["direct_harmonic"]=getattr(tp,"_direct_harmonic",False)
    plan["segmented_forward"]=getattr(tp,"_segmented_forward",False)
    plan["segment_neighbors"]=getattr(tp,"_segment_neighbors",32)
    plan["int32_index"]=getattr(tp,"_int32_message",False)
    plan["cached_edge_forward"]=getattr(tp,"_cached_edge_forward",False)
    plan["cached_edge_geometry"]=getattr(tp,"_cached_edge_geometry",False)
    generated_supported=tp.channels>0
    plan["generated_forward"]=getattr(tp,"_generated_forward",False) and generated_supported
    plan["generated_adjoint"]=getattr(tp,"_generated_adjoint",False) and generated_supported
    plan['generated_specialized_adjoint']=getattr(tp,'_generated_specialized_adjoint',False) and generated_supported
    plan["generated_layout"]=getattr(tp,"_generated_layout","packed") if generated_supported else "packed"
    plan["generated_blocks"]=getattr(tp,"_generated_blocks",0)
    plan["generated_schedule"]=getattr(tp,"_generated_schedule",0)
    plan["generated_reg_limit"]=getattr(tp,"_generated_reg_limit",0)
    plan["generated_channel_tile"]=getattr(tp,"_generated_channel_tile",0)
    plan['inference_adjoint_channel_tile']=getattr(tp,'_inference_adjoint_channel_tile',0) if not train_gate else 0
    plan['gate_adjoint_rows']=getattr(tp,'_inference_gate_adjoint_rows',0) if not train_gate else 0
    plan['gate_gemm_policy']=getattr(tp,'_inference_gemm_policy','torch') if not train_gate else 'torch'
    plan['gate_adjoint_edges']=getattr(tp,'_inference_message_adjoint_edges',0) if not train_gate else 0
    plan["generated_geometry_streaming"]=getattr(tp,"_generated_geometry_streaming",False)
    plan["generated_csr_forward"]=getattr(tp,"_generated_csr_forward",False)
    plan['generated_csr_fused_adjoint']=getattr(tp,'_generated_csr_fused_adjoint',False)
    plan['generated_topology']=getattr(tp,'_generated_topology','native')
    plan['fused_source_grouping']=getattr(tp,'_fused_source_grouping','degree')
    plan["csr_accumulator_limit"]=getattr(tp,"_csr_accumulator_limit",48)
    plan['csr_slice_search']=getattr(tp,'_csr_slice_search',False)
    plan["gate_cache_bytes"]=getattr(tp,"_gate_cache_bytes",0)
    plan['inference_degree_layout']=(getattr(tp,'_inference_degree_layout',False) and not train_gate
                                    and plan['generated_forward'] and plan['generated_adjoint'])
    plan['generated_second']=getattr(tp,'_generated_second',False)
    plan['generated_second_roles']=getattr(tp,'_generated_second_roles',False)
    plan['generated_second_channel_tile']=getattr(tp,'_generated_second_channel_tile',None)
    if train_gate:
        if not (plan['generated_forward'] and plan['generated_adjoint']):return None
        plan['gate_cache_bytes']=(1<<62) if tp._training_gate_schedule=='cached' else 0
    plan.update(getattr(tp,"_message_launch",{}))
    if not train_gate and plan['generated_adjoint'] and max(edge_attrs)<=3:
        from .k100_message_schedule import schedule
        plan=schedule(plan,sample)
    if output_nodes is not None:
        # Rectangular source/destination node sets are currently qualified for
        # generated frozen-weight inference. Other routes retain their fallback.
        if (train_gate or not plan['generated_forward'] or not plan['generated_adjoint']
                or plan['segmented_forward'] or plan['generated_csr_fused_adjoint']):
            return None
        if not 0 < output_nodes <= sample.shape[0]:
            raise ValueError('output_nodes must be a nonempty subset of input rows')
        plan['output_nodes']=output_nodes
    n=sample.shape[0];c=tp.channels
    x=torch.cat([x1[l].reshape(n,c*(2*l+1)) for l in sorted(x1)],-1).contiguous()
    sh=torch.cat([edge_attrs[l].squeeze(-2) for l in projection["keys"]],-1).contiguous()
    from .radial_checkpoint import radial_apply
    hidden=radial_apply(fc,edge_feats,hidden=True,checkpoint=getattr(tp,'_radial_checkpoint',False))
    bias=sample.new_empty(0) if last.bias is None else last.bias
    mask=sample.new_empty(0) if mask is None else mask.to(sample).reshape(e).contiguous()
    if (getattr(tp,"_sort_gate_edges",False) or plan["segmented_forward"]) and e:
        permutation=torch.argsort(dst,stable=True)
        sh=sh[permutation];hidden=hidden[permutation]
        src=src[permutation];dst=dst[permutation]
        if mask.numel():mask=mask[permutation]
    weight=tp.weight.to(sample).contiguous();gate_weight=last.weight
    if getattr(tp,"_fold_path_weights",False) and not train_gate:
        gate_weight=gate_weight*weight.reshape(-1,1)
        if bias.numel():bias=bias*weight.reshape(-1)
        weight=torch.ones_like(weight);plan["unit_weights"]=True
    if getattr(tp,'_generated_compact_cartesian',False):
        raise ValueError('Legacy STF-coordinate flag is retired; reconfigure the model with message_schedule=native or roles')
    operation=_StreamedGate
    if train_gate:
        from .training_gate_message import _TrainingGate
        operation=_TrainingGate
    output=operation.apply(x,sh,weight,hidden,gate_weight,bias,
                              src.contiguous(),dst.contiguous(),mask,plan,projection)
    from .fused_message import _PathBlocks
    return dict(enumerate(_PathBlocks.apply(output,plan)))
