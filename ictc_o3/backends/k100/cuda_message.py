"""Independent optional CUDA destination-CSR reduction, with live topology.

The forward and upstream-cotangent adjoint have unique output ownership.
Other first/second adjoints use our Triton kernels; higher derivatives use
native PyTorch recomputation. Compilation is lazy and stays in Torch's cache.
"""
from functools import lru_cache
from pathlib import Path

import torch
from ictc_o3.autograd.adjoint_requirements import required_gradients


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import CUDA_HOME
    from ictc_o3.backends.k100.native_jit import load
    source = Path(__file__).resolve().parent / "csrc"
    target_include = Path(CUDA_HOME or "") / "targets/x86_64-linux/include"
    includes = [str(target_include)] if CUDA_HOME and target_include.is_dir() else []
    return load(name="mace_ictc_csr_message_v1",
                sources=[str(source/"csr_message.cpp"),str(source/"csr_message.cu")],
                extra_include_paths=includes,
                extra_cflags=["-O3"],extra_cuda_cflags=["-O3"],verbose=False)


def _raw(x,b,w,g,src,mask,ptr,edges,p,incoming=None):
    original=(x,b,w,g)
    other=original if incoming is None else tuple(value.contiguous() if value is not None else base
        for value,base in zip(incoming,original))
    flags=0 if incoming is None else sum(1<<i for i,value in enumerate(incoming) if value is not None)
    return _extension().csr_message(x,b,w,g,*other,src,mask,ptr,edges,p["km"],
                                    p["c"],p["odim"],p["p"],incoming is not None,flags)


def _dx_raw(b,w,g,grad,dst,mask,ptr,edges,p,incoming=None):
    from ictc_o3.ops.fused_message import _channel_cotangent
    other=(b,w,g) if incoming is None else tuple(value.contiguous() if value is not None else base
        for value,base in zip(incoming[1:],(b,w,g)))
    flags=0 if incoming is None else sum(1<<i for i,value in enumerate(incoming) if value is not None)
    vc=_channel_cotangent(grad,p)
    blocks=[_extension().csr_dx(b,w,g,vc,*other,dst,mask,ptr,edges,layout,p["c"],ni,incoming is not None,flags)
            for _,ni,layout in p["dx_channel_groups"]]
    return torch.cat([value.reshape(grad.shape[0],p["c"]*ni) for value,(_,ni,_) in zip(blocks,p["dx_channel_groups"])],dim=1)


def _vjp(x,b,w,g,src,dst,mask,ptr,edges,grad,p,needed):
    from ictc_o3.ops.fused_message import _vjp_raw
    _,db,dw,dg=_vjp_raw(x,b,w,g,src,dst,mask,grad,p,(False,*needed[1:]))
    dx=_dx_raw(b,w,g,grad,dst,mask,ptr,edges,p) if needed[0] else None
    return dx,db,dw,dg


class _CSRAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,b,w,g,src,dst,mask,ptr,edges,srcptr,srcedges,grad,p,needed):
        ctx.plan=p;ctx.needed=needed;ctx.set_materialize_grads(False)
        grad=grad.contiguous();ctx.save_for_backward(x,b,w,g,src,dst,mask,ptr,edges,srcptr,srcedges,grad)
        return _vjp(x,b,w,g,src,dst,mask,srcptr,srcedges,grad,p,needed)

    @staticmethod
    def backward(ctx,*incoming):
        from ictc_o3.ops.fused_message import _native
        from ictc_o3.ops.fused_message import _second_raw
        x,b,w,g,src,dst,mask,ptr,edges,srcptr,srcedges,grad=ctx.saved_tensors
        p=ctx.plan
        *needs,need_grad=required_gradients((x,b,w,g,grad),(*ctx.needs_input_grad[:4],ctx.needs_input_grad[11]))
        if torch.is_grad_enabled():
            local=tuple(value.clone() for value in (x,b,w,g));cotangent=grad.clone()
            out=_native(*local,src,dst,mask,p)
            first=torch.autograd.grad(out,[value for value,need in zip(local,ctx.needed) if need],
                                      cotangent,create_graph=True,allow_unused=True)
            pairs=[(value,h) for value,h in zip(first,[v for v,need in zip(incoming,ctx.needed) if need])
                   if h is not None and value is not None and value.requires_grad]
            active=[value for value,need in zip((*local,cotangent),(*needs,need_grad)) if need]
            values=iter(torch.autograd.grad(tuple(a for a,_ in pairs),active,tuple(h for _,h in pairs),
                                           create_graph=True,allow_unused=True))
            dx,db,dw,dg,dy=[next(values) if need else None for need in (*needs,need_grad)]
        else:
            _,db,dw,dg,_=_second_raw(x,b,w,g,src,dst,mask,grad,p,incoming,(False,*needs[1:]),False)
            dx=_dx_raw(b,w,g,grad,dst,mask,srcptr,srcedges,p,incoming) if needs[0] else None
            dy=_raw(x,b,w,g,src,mask,ptr,edges,p,incoming) if need_grad else None
        return dx,db,dw,dg,None,None,None,None,None,None,None,dy,None,None


class _CSRMessage(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,b,w,g,src,dst,mask,p):
        edges=torch.argsort(dst,stable=True)
        counts=torch.bincount(dst,minlength=x.shape[0])
        ptr=torch.cat((counts.new_zeros(1),counts.cumsum(0)))
        if ctx.needs_input_grad[0]:
            srcedges=torch.argsort(src,stable=True);counts=torch.bincount(src,minlength=x.shape[0])
            srcptr=torch.cat((counts.new_zeros(1),counts.cumsum(0)))
        else:srcptr=ptr.new_empty(0);srcedges=src.new_empty(0)
        ctx.plan=p;ctx.save_for_backward(x,b,w,g,src,dst,mask,ptr,edges,srcptr,srcedges)
        return _raw(x,b,w,g,src,mask,ptr,edges,p)

    @staticmethod
    def backward(ctx,grad):
        x,b,w,g,src,dst,mask,ptr,edges,srcptr,srcedges=ctx.saved_tensors;p=ctx.plan
        needed=required_gradients((x,b,w,g),ctx.needs_input_grad[:4])
        if torch.is_grad_enabled():
            values=_CSRAdjoint.apply(x,b,w,g,src,dst,mask,ptr,edges,srcptr,srcedges,grad,p,needed)
        else:
            values=_vjp(x,b,w,g,src,dst,mask,srcptr,srcedges,grad,p,needed)
        return *values,None,None,None,None


def cuda_message(x,b,w,g,src,dst,mask,plan):
    return _CSRMessage.apply(x,b,w,g,src,dst,mask,plan)
