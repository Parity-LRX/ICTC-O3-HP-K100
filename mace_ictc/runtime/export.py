"""Prepared Triton operators that survive FX/export and lower into AOTI cubins.

Preparation is explicit and runs before tracing. The exported model needs no
Python or Triton runtime; torch.library.triton_op is lowered by AOTInductor.
Frozen deployment only: original parameters and checkpoint keys are retained.
"""
import importlib.util
import os
from pathlib import Path
import sys
from typing import Tuple

import torch

_SERIAL=0


def _sum(parts):
    values=list(parts)
    return " + ".join(f"({value})" for value in values) if values else "0.0"


def _message_source(p,projection):
    hd=projection["hdim"];c=p["c"];tiled=p.get("deployment_channel_tiles",False)
    halves=range(1 if tiled else (c+31)//32)
    canonical=p.get("generated_layout","packed")=="canonical"
    def input_index(xb,ni,i,half):
        channel=f"channel_base+{half*32}+lane" if tiled else f"{half*32}+lane"
        return f"{xb}+({channel})*{ni}+{i}" if canonical else f"{xb+i*c}+({channel})"
    def output_index(ob,m,mag,half):
        channel=f"channel_base+{half*32}+lane" if tiled else f"{half*32}+lane"
        return f"{ob}+({channel})*{m}+{mag}" if canonical else f"{ob+mag*c}+({channel})"
    def valid_channel(half):
        if tiled:return f"(valid & (channel_base+lane<{c}))"
        return "valid" if (half+1)*32<=c else f"(valid & (lane+{half*32}<{c}))"
    def path_index(path,half):
        base="channel_base+" if tiled else ""
        return f"{path*c+half*32}+{base}lane"
    columns=[[] for _ in range(p["bdim"])]
    for h,rows in enumerate(projection["static_rows"]):
        for col,a in rows:columns[col].append((h,a))
    lines=["import triton","import triton.language as tl","","@triton.jit",
        "def kernel(X,SH,W,G,S,D,M,V,O,DX,DH,DG,E,MODE:tl.constexpr,BLOCK:tl.constexpr):",
        "    row=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)",
        "    e=row//32;lane=row%32;valid=e<E",
        "    src=tl.load(S+e,valid,0);dst=tl.load(D+e,valid,0)",
        "    mask=tl.load(M+e,valid,0.)",
        "    channel_base=tl.program_id(1)*32" if tiled else "    channel_base=0"]
    def emit(value,indent=1):lines.append("    "*indent+value)
    for h in range(hd):
        emit(f"h{h}=tl.load(SH+e*{hd}+{h},valid,0.)")
        emit(f"dh{h}=tl.full((BLOCK,),0.,tl.float32)")
    groups={}
    for spec in p["path_specs"]:groups.setdefault((spec[2],spec[1]),[]).append(spec)
    for (xb,ni),paths in groups.items():
        for half in halves:
            for i in range(ni):
                emit(f"x{half}_{i}=tl.load(X+src*{p['xdim']}+{input_index(xb,ni,i,half)},{valid_channel(half)},0.)")
                emit(f"dx{half}_{i}=tl.full((BLOCK,),0.,tl.float32)")
        for path,ni,xb,bb,bs,ob,m in paths:
            for half in halves:
                emit(f"w{half}=tl.load(W+{path_index(path,half)},{valid_channel(half)},0.)")
                emit(f"g{half}=tl.load(G+e*{p['p']*c}+{path_index(path,half)},{valid_channel(half)},0.)")
            for i in range(ni):
                for mag in range(m):
                    values=columns[bb+i*bs+mag]
                    if values:emit(f"b{i}_{mag}="+_sum(f"{repr(a)}*h{h}" for h,a in values))
            emit("if MODE==0:")
            for half in halves:
                for mag in range(m):
                    value=_sum(f"x{half}_{i}*b{i}_{mag}" for i in range(ni) if columns[bb+i*bs+mag])
                    emit(f"y=(({value})*w{half})*g{half}*mask",2)
                    emit(f"tl.atomic_add(O+dst*{p['odim']}+{output_index(ob,m,mag,half)},y,{valid_channel(half)},sem='relaxed')",2)
            emit("else:")
            for half in halves:
                for mag in range(m):emit(f"v{half}_{mag}=tl.load(V+dst*{p['odim']}+{output_index(ob,m,mag,half)},{valid_channel(half)},0.)*mask",2)
                for i in range(ni):
                    value=_sum(f"b{i}_{mag}*v{half}_{mag}" for mag in range(m) if columns[bb+i*bs+mag])
                    emit(f"z{half}_{i}={value}",2)
                    emit(f"dx{half}_{i}=dx{half}_{i}+z{half}_{i}*(w{half}*g{half})",2)
                value=_sum(f"x{half}_{i}*z{half}_{i}" for i in range(ni))
                emit(f"tl.store(DG+e*{p['p']*c}+{path_index(path,half)},({value})*w{half},{valid_channel(half)})",2)
            for i in range(ni):
                for mag in range(m):
                    values=columns[bb+i*bs+mag]
                    if not values:continue
                    value="+".join(f"x{half}_{i}*(v{half}_{mag}*(g{half}*w{half}))" for half in halves)
                    emit(f"db={value}",2)
                    for h,a in values:emit(f"dh{h}=dh{h}+{repr(a)}*db",2)
        emit("if MODE==1:")
        for half in halves:
            for i in range(ni):emit(f"tl.atomic_add(DX+src*{p['xdim']}+{input_index(xb,ni,i,half)},dx{half}_{i},{valid_channel(half)},sem='relaxed')",2)
    emit("if MODE==1:")
    emit("edge=tl.program_id(0).to(tl.int64)*(BLOCK//32)+tl.arange(0,BLOCK//32)",2)
    for h in range(hd):
        value=f"DH+edge*{hd}+{h},tl.sum(tl.reshape(dh{h},(BLOCK//32,32)),1),edge<E"
        emit(f"tl.atomic_add({value},sem='relaxed')" if p.get("deployment_partitioned",False) or tiled else f"tl.store({value})",2)
    return "\n".join(lines)+"\n"


def prepare_message(layer,device):
    from ictc_o3.ops.fused_message import _plan
    from ictc_o3.ops.fused_message import _channel_node
    from ictc_o3.ops.fused_message import _channel_cotangent
    from ictc_o3.ops.fused_message import _normal_message_output
    from ictc_o3.ops.fused_message import _normal_feature_gradient
    from ictc_o3.ops.bond_projection import projection_plan
    from torch.library import triton_op,wrap_triton
    import triton
    tp=layer.tp;p=dict(_plan(tp,list(range(layer.input_lmax+1)),device,True))
    p["generated_layout"]=getattr(tp,"_generated_layout","packed")
    canonical=p["generated_layout"]=="canonical"
    projection=projection_plan(tp,tp._get_bond_projectors(device,torch.float32),device,torch.float32)
    high=4<=int(layer.tp.lmax)<=5
    p["deployment_channel_tiles"]=high and p["c"]>=32
    parts=[]
    if high:
        # Bound compiler/register work by input degree, while retaining the
        # same Cartesian paths, graph aggregation and joint first adjoint.
        groups={}
        for row in p["path_specs"]:groups.setdefault(row[2],[]).append(row)
        for rows in groups.values():
            # A high Cartesian degree has dense projector coefficients. An
            # input-degree partition alone can still exceed 500 KiB of source
            # and stall Triton SSA scheduling. Bound literal expansion by
            # paths as well; DX/DH already use exact graph atomic reductions.
            for start in range(0,len(rows),2):
                part=dict(p,path_specs=rows[start:start+2],deployment_partitioned=True)
                parts.append(part)
    else:parts=[p]
    kernels=[_module(_message_source(part,projection),"message").kernel for part in parts]
    hd=projection["hdim"]
    tiles=triton.cdiv(p["c"],32) if p["deployment_channel_tiles"] else 1

    @triton_op(_name("message_backward"),mutates_args={})
    def backward_op(x:torch.Tensor,sh:torch.Tensor,w:torch.Tensor,g:torch.Tensor,s:torch.Tensor,d:torch.Tensor,m:torch.Tensor,v:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        e=s.shape[0];xc=x if canonical else _channel_node(x,p);vc=v if canonical else _channel_cotangent(v,p)
        dx=torch.zeros_like(x);dh=torch.zeros_like(sh) if high else torch.empty_like(sh);dg=torch.empty_like(g)
        for kernel in kernels:
            wrap_triton(kernel)[(triton.cdiv(e*32,128),tiles)](xc,sh,w,g,s,d,m,vc,vc,dx,dh,dg,e,1,128,num_warps=4,enable_fp_fusion=True)
        return (dx if canonical else _normal_feature_gradient(dx,p)),dh,dg

    @triton_op(_name("message"),mutates_args={})
    def forward_op(x:torch.Tensor,sh:torch.Tensor,w:torch.Tensor,g:torch.Tensor,s:torch.Tensor,d:torch.Tensor,m:torch.Tensor)->torch.Tensor:
        e=s.shape[0];xc=x if canonical else _channel_node(x,p);out=x.new_zeros(x.shape[0],p["odim"])
        for kernel in kernels:
            wrap_triton(kernel)[(triton.cdiv(e*32,128),tiles)](xc,sh,w,g,s,d,m,out,out,xc,sh,g,e,0,128,num_warps=4,enable_fp_fusion=True)
        return out if canonical else _normal_message_output(out,p)

    def setup(ctx,inputs,output):ctx.save_for_backward(*inputs)
    def backward(ctx,v):
        x,sh,w,g,s,d,m=ctx.saved_tensors
        dx,dh,dg=backward_op(x,sh,w,g,s,d,m,v.contiguous())
        return dx,dh,None,dg,None,None,None
    forward_op.register_autograd(backward,setup_context=setup)
    def evaluate(x1,edge_attrs,gates,s,d,mask):
        n=next(iter(x1.values())).shape[0];e=s.shape[0]
        x=torch.cat([x1[l].reshape(n,p["c"]*(2*l+1)) for l in sorted(x1)],-1).contiguous()
        sh=torch.cat([edge_attrs[l].squeeze(-2) for l in projection["keys"]],-1).contiguous()
        mask=x.new_ones(e) if mask is None else mask.to(x).reshape(e).contiguous()
        out=forward_op(x,sh,tp.weight,gates.reshape(e,p["p"],p["c"]).contiguous(),s.contiguous(),d.contiguous(),mask)
        return {l:out[:,p["ooffset"][l]:p["ooffset"][l]+p["c"]*p["counts"].get(l,0)*(2*l+1)].reshape(n,p["c"]*p["counts"].get(l,0),2*l+1) for l in range(tp.lmax+1)}
    layer._deployment_message=evaluate
    layer._deployment_message_kind="degree_partitioned" if high else "unrolled"


def _name(stem):
    global _SERIAL
    _SERIAL+=1
    return f"mace_ictc_deploy::{stem}_{_SERIAL}"


def _module(source,stem):
    global _SERIAL
    _SERIAL+=1
    root=Path(os.environ.get("TRITON_CACHE_DIR",str(Path.home()/".triton/cache")))/"deployment"
    root.mkdir(parents=True,exist_ok=True)
    name=f"ictc_deploy_{stem}_{os.getpid()}_{_SERIAL}";path=root/f"{name}.py"
    path.write_text(source)
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value)
    return value


def prepare_harmonics(model,device):
    from ictc_o3.ops.generated_direction_harmonics import _plan
    from torch.library import triton_op,wrap_triton
    import triton
    lmax=int(model.ictd_fix_edge_lmax)
    plan=_plan(torch.empty(1,3,device=device,dtype=torch.float32),lmax)
    kernel=plan["kernel"];hd=plan["hdim"]

    @triton_op(_name("harmonic_backward"),mutates_args={})
    def backward_op(n:torch.Tensor,g:torch.Tensor)->torch.Tensor:
        n=n.contiguous();g=g.contiguous();dn=torch.empty_like(n);e=n.shape[0]
        wrap_triton(kernel)[(triton.cdiv(e,256),)](n,g,n,n,dn,n,e,1,True,False,256,num_warps=4,enable_fp_fusion=True)
        return dn

    @triton_op(_name("harmonic"),mutates_args={})
    def forward_op(n:torch.Tensor)->torch.Tensor:
        n=n.contiguous();e=n.shape[0];out=n.new_empty(e,hd)
        wrap_triton(kernel)[(triton.cdiv(e,256),)](n,n,n,out,n,n,e,0,False,False,256,num_warps=4,enable_fp_fusion=True)
        return out

    def setup(ctx,inputs,output):ctx.save_for_backward(inputs[0])
    def backward(ctx,g):return backward_op(ctx.saved_tensors[0],g)
    forward_op.register_autograd(backward,setup_context=setup)
    def evaluate(n):return list(forward_op(n).split([2*l+1 for l in range(lmax+1)],-1))
    model._deployment_harmonics=evaluate


def prepare_contraction(owner,device):
    from ictc_o3.ops.factorized_contraction import _plan
    c=int(owner.channels);size=(int(owner.lmax)+1)**2
    plan=_plan(owner,torch.empty(1,c,size,device=device,dtype=torch.float32))
    if plan is None:raise ValueError("This symmetric contraction has no qualified factorized deployment plan")
    banks=[]
    with torch.no_grad():
        for module,order,projection in plan["projections"]:
            weights=[module.weights_max,*module.weights]
            banks.append(torch.einsum("rp,zpc->zcr",projection,weights[module.correlation-order]))
        bank=torch.cat(banks,-1).contiguous()
    owner.register_buffer("_deployment_bank",bank,persistent=False)
    owner.register_buffer("_deployment_q",plan["q"],persistent=False)
    for l,q in enumerate(plan["qout"]):owner.register_buffer(f"_deployment_qout_{l}",q,persistent=False)
    orders={module.correlation for module in owner.symmetric_contractions.contractions}
    tail_quadratic=(orders=={2} and owner.target_lmax>0 and c%32!=0)
    if 4<=owner.lmax<=5 and (orders=={3} or tail_quadratic):
        from .deployment_sparse import prepare_sparse_contraction
        owner._deployment_contraction=prepare_sparse_contraction(owner,plan,_name)
        owner._deployment_contraction_kind="sparse"
        return
    forward_op=_prepare_circuit(plan,c,size)

    def evaluate(x,idx):
        native=x@owner._deployment_q
        value=forward_op(native,owner._deployment_bank,idx.contiguous())
        return torch.cat([(part@getattr(owner,f"_deployment_qout_{l}").T).reshape(x.shape[0],-1)
                          for l,part in enumerate(value.split(plan["sizes"],-1))],-1)
    owner._deployment_contraction=evaluate
    owner._deployment_contraction_kind="circuit"


def _prepare_circuit(plan,c,size):
    from ictc_o3.ops.factorized_circuit import _generate
    from torch.library import triton_op,wrap_triton
    import triton
    p=plan["final"]["sizes"][2];m=plan["final"]["sizes"][3]
    source=_generate(plan["pair"],plan["final"],plan["input_size"])
    source=source.replace("def circuit(X,A,G,H,HA,Y,N,","def circuit(X,A,G,H,HA,S,Y,N,")
    source=source.replace("tl.load(A+c*",f"tl.load(A+tl.load(S+n,valid,0).to(tl.int64)*C*{p}+c*")
    kernel=_module(source,"circuit").circuit

    @triton_op(_name("contraction_backward"),mutates_args={})
    def backward_op(x:torch.Tensor,a:torch.Tensor,s:torch.Tensor,g:torch.Tensor)->torch.Tensor:
        n=x.shape[0];packed=x.transpose(1,2).contiguous();gp=g.transpose(1,2).contiguous()
        out=x.new_empty(n,size,c)
        wrap_triton(kernel)[(triton.cdiv(n*c,128),)](packed,a,gp,packed,a,s,out,n,c,1,False,False,128,num_warps=4,enable_fp_fusion=True)
        return out.transpose(1,2).contiguous()

    @triton_op(_name("contraction"),mutates_args={})
    def forward_op(x:torch.Tensor,a:torch.Tensor,s:torch.Tensor)->torch.Tensor:
        n=x.shape[0];packed=x.transpose(1,2).contiguous();out=x.new_empty(n,m,c)
        wrap_triton(kernel)[(triton.cdiv(n*c,128),)](packed,a,packed,packed,a,s,out,n,c,0,False,False,128,num_warps=4,enable_fp_fusion=True)
        return out.transpose(1,2).contiguous()

    def setup(ctx,inputs,output):ctx.save_for_backward(*inputs)
    def backward(ctx,g):
        x,a,s=ctx.saved_tensors
        return backward_op(x,a,s,g),None,None
    forward_op.register_autograd(backward,setup_context=setup)
    return forward_op


def configure_mace_deployment_plan(model, *, policy="auto", generated_layout=None,
                                   reference_buffer_policy="auto"):
    """Default to generated inference for qualified models; preserve other models."""
    from mace_ictc.runtime.planning import (
        generated_plan_unavailable, mace_backbone_core, set_mace_backbone_plan,
    )
    if policy not in ("auto", "checkpoint", "generated"):
        raise ValueError("deployment operator plan must be auto, checkpoint or generated")
    if generated_layout not in (None,"packed","canonical","fused_pack"):
        raise ValueError("Unsupported generated deployment layout")
    if reference_buffer_policy not in ("auto","device"):
        raise ValueError("reference_buffer_policy must be auto or device")
    from ictc_o3.ops.reference_u_residency import restore_model_reference_u
    if policy == "checkpoint":
        restore_model_reference_u(model)
        return {"requested": policy, "selected": "checkpoint", "reasons": [], "counts": {}}
    core = mace_backbone_core(model)
    reasons = generated_plan_unavailable(core)
    if not hasattr(torch.library, "triton_op"):
        reasons.append("PyTorch with torch.library.triton_op is required")
    if reasons:
        if policy == "generated":
            raise ValueError("Generated deployment unavailable: " + "; ".join(reasons))
        restore_model_reference_u(model)
        return {"requested": policy, "selected": "checkpoint", "reasons": reasons, "counts": {}}
    from ictc_o3.backends.k100.fp32_policy import strict_fp32
    strict_fp32()
    layouts=[getattr(layer.tp,"_generated_layout","packed") if generated_layout is None else generated_layout
             for layer in core.interactions]
    set_mace_backbone_plan(core, shared_channel_gemm=True)
    for layer,layout in zip(core.interactions,layouts):layer.tp._generated_layout=layout
    from ictc_o3.ops.reference_u_residency import host_reference_unavailable
    counts = prepare_deployment_triton(core,host_reference=(reference_buffer_policy=="auto" and host_reference_unavailable(model) is None))
    return {"requested": policy, "selected": "generated", "reasons": [], "counts": counts,
            "reference_buffer_policy":reference_buffer_policy}


def prepare_deployment_triton(model,*,host_reference=True):
    """Prepare an independent parameterized FP32 model before FX/export."""
    from ictc_o3.ops.symmetric_contraction import ICTDBridgeUSymmetricContractionSO3
    if not hasattr(torch.library,"triton_op"):raise RuntimeError("Deployment kernels require torch.library.triton_op")
    sample=next(model.parameters())
    if sample.dtype!=torch.float32 or not sample.is_cuda:raise ValueError("Deployment Triton mode requires CUDA FP32")
    if int(model.channels)<1 or int(model.ictd_fix_edge_lmax)<0:raise ValueError("Deployment Triton mode requires positive channels and a nonnegative degree")
    if getattr(model,"angular_basis","ictd")!="ictd":raise ValueError("Generated deployment requires the Cartesian ICTC angular basis")
    if any(int(getattr(layer,"interaction_attn_heads",0)) for layer in model.interactions):
        raise ValueError("Generated MACE deployment does not support interaction attention")
    if any(value.requires_grad for value in model.parameters()):raise ValueError("Freeze all model parameters before preparing deployment")
    if any(type(module).__module__.startswith("cuequivariance") or getattr(module,"tp_cueq",None) is not None for module in model.modules()):raise ValueError("Independent deployment cannot mix cuEq")
    prepare_harmonics(model,sample.device)
    for layer in model.interactions:prepare_message(layer,sample.device)
    from ictc_o3.ops.reference_u_residency import host_reference_unavailable
    from ictc_o3.ops.reference_u_residency import offload_reference_u
    from ictc_o3.ops.reference_u_residency import restore_reference_u
    count=sparse_count=host_count=0
    host_allowed=host_reference and host_reference_unavailable(model) is None
    for module in model.modules():
        if isinstance(module,ICTDBridgeUSymmetricContractionSO3):
            prepare_contraction(module,sample.device);count+=1
            sparse=getattr(module,"_deployment_contraction_kind",None)=="sparse"
            sparse_count+=int(sparse)
            cubic=all(c.correlation==3 for c in module.symmetric_contractions.contractions)
            module._host_reference_u=host_allowed and sparse and cubic
            if module._host_reference_u:
                offload_reference_u(module,sample);host_count+=1
            else:restore_reference_u(module,sample.device)
    if count!=len(model.interactions):raise ValueError("Generated deployment requires bridge-U symmetric products in every interaction")
    return {"harmonics":1,"factorized_contractions":count,"messages":len(model.interactions),
            "high_l_sparse_sc":sparse_count,
            "high_l_partitioned_messages":sum(layer._deployment_message_kind=="degree_partitioned" for layer in model.interactions),
            "host_reference_sc":host_count}
