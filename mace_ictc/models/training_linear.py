"""Bounded-row BLAS adjoints for a qualified K100 radial projection.

The forward remains the original linear map. First and mixed adjoints use
the same live parameters and BLAS operands, with bounded row reductions.
"""
import torch
import torch.nn.functional as F
from .adjoint_requirements import required_gradients


def _feature_raw(g,w,rows):
    g=g.contiguous();w=w.contiguous()
    out=g.new_empty((g.shape[0],w.shape[1]))
    for start in range(0,g.shape[0],rows):
        torch.mm(g[start:start+rows],w,out=out[start:start+rows])
    return out


def _weight_raw(g,x,rows):
    g=g.contiguous();x=x.contiguous()
    out=g.new_zeros((g.shape[1],x.shape[1]))
    for start in range(0,g.shape[0],rows):
        if start==0:torch.mm(g[start:start+rows].T,x[start:start+rows],out=out)
        else:out.addmm_(g[start:start+rows].T,x[start:start+rows])
    return out


class _FeatureAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx,g,w,rows):
        ctx.save_for_backward(g,w);ctx.rows=rows
        return _feature_raw(g,w,rows)

    @staticmethod
    def backward(ctx,h):
        g,w=ctx.saved_tensors
        ng,nw=required_gradients((g,w),ctx.needs_input_grad[:2])
        dg=F.linear(h,w) if ng else None
        dw=((g.T@h) if torch.is_grad_enabled() else _weight_raw(g,h,ctx.rows)) if nw else None
        return dg,dw,None


class _WeightAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx,g,x,rows):
        ctx.save_for_backward(g,x);ctx.rows=rows
        return _weight_raw(g,x,rows)

    @staticmethod
    def backward(ctx,h):
        g,x=ctx.saved_tensors
        ng,nx=required_gradients((g,x),ctx.needs_input_grad[:2])
        dg=F.linear(x,h) if ng else None
        dx=((g@h) if torch.is_grad_enabled() else _feature_raw(g,h,ctx.rows)) if nx else None
        return dg,dx,None


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,bias,rows):
        ctx.save_for_backward(x,w,bias);ctx.rows=rows
        return F.linear(x,w,bias)

    @staticmethod
    def backward(ctx,g):
        x,w,bias=ctx.saved_tensors
        nx,nw,nb=required_gradients((x,w,bias),ctx.needs_input_grad[:3])
        if torch.is_grad_enabled():
            dx=_FeatureAdjoint.apply(g,w,ctx.rows) if nx else None
            dw=_WeightAdjoint.apply(g,x,ctx.rows) if nw else None
        else:
            dx=_feature_raw(g,w,ctx.rows) if nx else None
            dw=_weight_raw(g,x,ctx.rows) if nw else None
        return dx,dw,g.sum(0) if nb else None,None


def tiled_linear(x,w,bias=None,*,rows):
    """Internal explicit layout, also used for analytic adjoint regression."""
    if not isinstance(rows,int) or isinstance(rows,bool) or rows<1:
        raise ValueError('rows must be a positive integer')
    return _Linear.apply(x,w,bias,rows)


def gate_linear(x,w,bias=None,*,policy):
    """Keep unmeasured shapes/stacks on their original PyTorch linear."""
    qualified=(policy=='k100_tiled' and x.ndim==2 and w.shape==(704,64)
        and x.shape[1]==64 and x.shape[0]>=65536 and torch.version.hip=='6.3.26093'
        and torch.__version__=='2.7.1' and x.is_cuda and x.dtype==torch.float32
        and w.is_contiguous() and w.device==x.device and w.dtype==x.dtype)
    if qualified:
        properties=torch.cuda.get_device_properties(x.device)
        qualified=getattr(properties,'gcnArchName','').split(':')[0]=='gfx926'
    return tiled_linear(x,w,bias,rows=16384) if qualified else F.linear(x,w,bias)
