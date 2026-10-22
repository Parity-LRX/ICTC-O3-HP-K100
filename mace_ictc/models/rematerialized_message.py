"""Save harmonic geometry instead of its expanded edge projection.

The linear geometry map is rebuilt only when a message adjoint needs it.
First and mixed second gradients reduce back to the harmonic input; no
expanded geometry/cotangent is retained in the differentiable graph.
"""
import torch
from .adjoint_requirements import required_gradients
from .bond_projection import _forward_raw as project, _transpose_raw as transpose


def _native(x,sh,w,g,src,dst,mask,p,projection):
    from .fused_message import _native as native_message
    from .fused_message import _normal_feature_gradient,_channel_cotangent
    packed=bool(p.get('native_packed',False))
    if packed:x=_normal_feature_gradient(x,p)
    magnetic=torch.cat([sh[:,offset:offset+size]@matrix
                        for offset,size,matrix in projection["groups"]],-1)
    value=native_message(x,magnetic,w,g,src,dst,mask,p)
    return _channel_cotangent(value,p) if packed else value


def _first(x,sh,w,g,src,dst,mask,grad,p,projection,needed):
    if p.get('training_adjoint_channel_tile') is not None:
        p={**p,'generated_channel_tile':p['training_adjoint_channel_tile']}
    from .fused_message import _vjp_raw,_normal_feature_gradient,_channel_inputs
    dx=torch.zeros_like(x) if needed[0] else None
    dsh=torch.empty_like(sh) if needed[1] else None
    dw=torch.zeros_like(w) if needed[2] else None
    dg=torch.empty_like(g) if needed[3] else None
    canonical=(p.get("generated_layout","packed")=="canonical"
               and p.get("generated_adjoint",False) and not needed[2])
    native_packed=bool(p.get('native_packed',False))
    packed=((x,grad) if canonical or native_packed else _channel_inputs(x,grad,p)) if src.numel() else None
    fused_source=bool(p.get('generated_csr_fused_adjoint',False) and p.get('generated_adjoint',False) and not needed[2])
    if fused_source:
        from .generated_csr_forward import topology
        from .generated_csr_fused_adjoint import adjoint
        csr=topology(src,x.shape[0],p.get('generated_topology','native'))
        if dsh is not None:dsh.zero_()
        for start,end in _tiles(src.numel(),p):
            adjoint(packed[0],sh[start:end],w,g[start:end],dst[start:end],_edge_slice(mask,start,end),
                packed[1],dx,dsh[start:end] if dsh is not None else None,
                dg[start:end] if dg is not None else None,p,projection,csr,start,(needed[0],needed[1],needed[3]))
        if dx is not None and not (canonical or native_packed):dx=_normal_feature_gradient(dx,p)
        return dx,dsh,dw,dg
    csr=None
    if needed[0] and p.get('generated_csr_adjoint',False) and p.get('generated_adjoint',False) and not needed[2]:
        from .generated_csr_forward import topology
        csr=topology(src,x.shape[0],p.get('generated_topology','native'))
    for start,end in _tiles(src.numel(),p):
        if p.get("generated_adjoint",False) and not needed[2]:
            if p.get('generated_triton_groups',False) and not needed[1]:
                from .generated_triton_message import adjoint
                adjoint(packed[0],sh[start:end],w,g[start:end],src[start:end],dst[start:end],
                    _edge_slice(mask,start,end),packed[1],dx,dg[start:end] if dg is not None else None,
                    p,projection,(needed[0],needed[3]))
                continue
            from .generated_cartesian_message import generated_adjoint
            if csr is not None:
                from .generated_csr_adjoint import source_adjoint
                source_adjoint(sh[start:end],w,g[start:end],dst[start:end],_edge_slice(mask,start,end),
                    packed[1],dx,p,projection,csr,start)
            ds,part_dg=generated_adjoint(packed[0],sh[start:end],w,g[start:end],src[start:end],dst[start:end],
                _edge_slice(mask,start,end),packed[1],dx,p,projection,(needed[0] and csr is None,needed[1],needed[3]),
                harmonic_output=dsh[start:end] if dsh is not None else None,
                gate_output=dg[start:end] if dg is not None else None)
            continue
        magnetic=project(sh[start:end],projection)
        _,db,part_dw,_=_vjp_raw(x,magnetic,w,g[start:end],src[start:end],dst[start:end],
            _edge_slice(mask,start,end),grad,p,needed,feature_output=dx,
            gate_output=dg[start:end] if dg is not None else None,channel_inputs=packed,
            harmonic_output=dsh[start:end] if dsh is not None else None,projection=projection)
        if db is not None:dsh[start:end].copy_(transpose(db,projection))
        if part_dw is not None:dw.add_(part_dw)
        del magnetic,db
    if p.get('release_packed_inputs',False):del packed
    if dx is not None and not (canonical or native_packed):dx=_normal_feature_gradient(dx,p)
    return dx,dsh,dw,dg


def _edge_slice(value,start,end):
    return value[start:end] if value is not None and value.numel() else value


def _tiles(edges,p):
    size=p.get("edge_chunk_size",65536)
    return ((start,min(start+size,edges)) for start in range(0,edges,size))


def _second(x,sh,w,g,src,dst,mask,grad,p,projection,incoming,needed,need_grad):
    from .fused_message import _second_raw,_normal_feature_gradient,_normal_message_output,_channel_inputs,_channel_node
    if not any(value is not None for value in incoming):return (None,)*5
    if p.get("generated_second",False) and not needed[2] and incoming[2] is None:
        from .generated_message_second import generated_second
        return generated_second(x,sh,w,g,src,dst,mask,grad,p,projection,incoming,needed,need_grad)
    if p.get('native_packed',False):
        from .fused_message import _channel_cotangent
        hx,hsh,hw,hg=incoming
        values=_second(_normal_feature_gradient(x,p),sh,w,g,src,dst,mask,_normal_message_output(grad,p),
            {**p,'native_packed':False},projection,
            (_normal_feature_gradient(hx,p) if hx is not None else None,hsh,hw,hg),needed,need_grad)
        dx,dsh,dw,dg,dy=values
        return _channel_node(dx,p) if dx is not None else None,dsh,dw,dg,_channel_cotangent(dy,p) if dy is not None else None
    hx,hsh,hw,hg=incoming
    dx=torch.zeros_like(x) if needed[0] else None
    dsh=torch.empty_like(sh) if needed[1] else None
    dw=torch.zeros_like(w) if needed[2] else None
    dg=torch.empty_like(g) if needed[3] else None
    dy=x.new_zeros(x.shape[0],p["odim"]) if need_grad else None
    if src.numel():
        xc,vc=_channel_inputs(x,grad,p);hxc=_channel_node(hx,p) if hx is not None else xc
        packed=(xc,vc,hxc)
    else:packed=None
    for start,end in _tiles(src.numel(),p):
        magnetic=project(sh[start:end],projection)
        hb=project(hsh[start:end],projection) if hsh is not None else None
        _,db,part_dw,_,_=_second_raw(x,magnetic,w,g[start:end],src[start:end],dst[start:end],
            _edge_slice(mask,start,end),grad,p,(hx,hb,hw,_edge_slice(hg,start,end)),needed,need_grad,
            feature_output=dx,gate_output=dg[start:end] if dg is not None else None,cotangent_output=dy,
            channel_inputs=packed)
        if db is not None:dsh[start:end].copy_(transpose(db,projection))
        if part_dw is not None:dw.add_(part_dw)
        del magnetic,hb,db
    if dx is not None:dx=_normal_feature_gradient(dx,p)
    if dy is not None:dy=_normal_message_output(dy,p)
    return dx,dsh,dw,dg,dy


class _Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,sh,w,g,src,dst,mask,grad,p,projection,needed):
        ctx.set_materialize_grads(False);ctx.plan=p;ctx.projection=projection;ctx.needed=needed
        grad=grad.contiguous();ctx.save_for_backward(x,sh,w,g,src,dst,mask,grad)
        return _first(x,sh,w,g,src,dst,mask,grad,p,projection,needed)

    @staticmethod
    def backward(ctx,hx,hsh,hw,hg):
        x,sh,w,g,src,dst,mask,grad=ctx.saved_tensors;p=ctx.plan;projection=ctx.projection
        *needs,need_grad=required_gradients((x,sh,w,g,grad),(*ctx.needs_input_grad[:4],ctx.needs_input_grad[7]))
        incoming=(hx,hsh,hw,hg)
        if torch.is_grad_enabled():
            local=tuple(value.clone() for value in (x,sh,w,g));cotangent=grad.clone()
            out=_native(*local,src,dst,mask,p,projection)
            first=torch.autograd.grad(out,[value for value,need in zip(local,ctx.needed) if need],
                        cotangent,create_graph=True,allow_unused=True)
            pairs=[(value,h) for value,h in zip(first,[v for v,need in zip(incoming,ctx.needed) if need])
                   if h is not None and value is not None and value.requires_grad]
            active=[value for value,need in zip((*local,cotangent),(*needs,need_grad)) if need]
            values=iter(torch.autograd.grad(tuple(a for a,_ in pairs),active,tuple(h for _,h in pairs),
                                create_graph=True,allow_unused=True))
            dx,dsh,dw,dg,dy=[next(values) if need else None for need in (*needs,need_grad)]
        else:
            dx,dsh,dw,dg,dy=_second(x,sh,w,g,src,dst,mask,grad,p,projection,incoming,needs,need_grad)
        return dx,dsh,dw,dg,None,None,None,dy,None,None,None


class _Message(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,sh,w,g,src,dst,mask,p,projection):
        from .fused_message import _group_message_raw,_normal_message_output,_channel_node
        ctx.plan=p;ctx.projection=projection;ctx.save_for_backward(x,sh,w,g,src,dst,mask)
        out=x.new_zeros(x.shape[0],p["odim"])
        canonical=p.get("generated_layout","packed")=="canonical" and p.get("generated_forward",False)
        native_packed=bool(p.get('native_packed',False))
        xc=x if canonical or native_packed else _channel_node(x,p)
        if src.numel() and p.get('generated_csr_fused_adjoint',False) and not w.requires_grad and any(v.requires_grad for v in (x,sh,g)):
            from .generated_csr_fused_adjoint import _extension
            _extension(p,projection)
        csr=None
        from .generated_csr_forward import enabled
        if enabled(p.get("generated_csr_forward",False),src.numel(),x.shape[0]):
            from .generated_csr_forward import topology
            csr=topology(dst,x.shape[0],p.get('generated_topology','native'),p.get('csr_slice_search',False))
        for start,end in _tiles(src.numel(),p):
            if p.get('generated_triton_groups',False) and csr is None:
                from .generated_triton_message import forward
                forward(xc,sh[start:end],w,g[start:end],src[start:end],dst[start:end],
                        _edge_slice(mask,start,end),out,p,projection)
                continue
            if csr is not None:
                from .generated_csr_forward import forward
                forward(xc,sh[start:end],w,g[start:end],src[start:end],_edge_slice(mask,start,end),out,p,projection,csr,start,dst[start:end])
                continue
            if p.get("generated_forward",False):
                from .generated_cartesian_message import generated_forward
                generated_forward(xc,sh[start:end],w,g[start:end],src[start:end],dst[start:end],
                    _edge_slice(mask,start,end),out,p,projection)
                continue
            magnetic=project(sh[start:end],projection)
            _group_message_raw(x,magnetic,w,g[start:end],src[start:end],dst[start:end],
                               _edge_slice(mask,start,end),p,output=out,channel_node=xc)
            del magnetic
        if p.get('release_packed_inputs',False):del xc
        return out if canonical or native_packed else _normal_message_output(out,p)

    @staticmethod
    def backward(ctx,grad):
        x,sh,w,g,src,dst,mask=ctx.saved_tensors;grad=grad.contiguous()
        needed=required_gradients((x,sh,w,g),ctx.needs_input_grad[:4])
        if torch.is_grad_enabled():
            values=_Adjoint.apply(x,sh,w,g,src,dst,mask,grad,ctx.plan,ctx.projection,needed)
        else:values=_first(x,sh,w,g,src,dst,mask,grad,ctx.plan,ctx.projection,needed)
        return *values,None,None,None,None,None


def rematerialized_message(x,sh,w,g,src,dst,mask,p,projection):
    return _Message.apply(x,sh,w,g,src,dst,mask,p,projection)
