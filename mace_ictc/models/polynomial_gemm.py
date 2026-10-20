"""Rematerialized polynomial GEMM and its exact mixed second adjoint.

Save features, coefficients and node cotangents rather than O(N*C*T)
monomial/cotangent tensors. Each temporary is rebuilt from live inputs.
"""
import torch

from .monomial_operator import _native, _value_raw, _adjoint_raw, monomials, triton
from .adjoint_requirements import required_gradients


def _first(x,coeff,grad,indices,plan,needed):
    if coeff.shape[-1]==1:
        from .scalar_polynomial import scalar_first
        return scalar_first(x,coeff,grad,indices,plan,needed)
    dx=dc=None
    if needed[0]:
        q=torch.bmm(grad,coeff.transpose(1,2))
        dx=_adjoint_raw(x,q,plan,indices.shape[0])
        del q
    if needed[1]:
        phi=_value_raw(x,indices)
        dc=torch.bmm(phi.transpose(1,2),grad)
    return dx,dc


class _GemmAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,coeff,grad,indices,plan,needed):
        ctx.needed=needed;ctx.set_materialize_grads(False)
        grad=grad.contiguous();ctx.save_for_backward(x,coeff,grad,indices,plan)
        return _first(x,coeff,grad,indices,plan,needed)

    @staticmethod
    def backward(ctx,hx,hc):
        x,coeff,grad,indices,plan=ctx.saved_tensors
        needs=required_gradients((x,coeff,grad),ctx.needs_input_grad[:3])
        if torch.is_grad_enabled():
            local=tuple(value.clone() for value in (x,coeff,grad))
            out=torch.bmm(_native(local[0],indices),local[1])
            first=torch.autograd.grad(out,[value for value,need in zip(local[:2],ctx.needed) if need],
                                      local[2],create_graph=True,allow_unused=True)
            pairs=[(value,h) for value,h in zip(first,[v for v,need in zip((hx,hc),ctx.needed) if need])
                   if h is not None and value is not None and value.requires_grad]
            if pairs:
                active=[value for value,need in zip(local,needs) if need]
                values=iter(torch.autograd.grad(tuple(a for a,_ in pairs),active,tuple(h for _,h in pairs),
                                               create_graph=True,allow_unused=True))
                dx,dc,dy=[next(values) if need else None for need in needs]
            else:
                dx,dc,dy=[value*0 if need else None for value,need in zip(local,needs)]
        else:
            if coeff.shape[-1]==1:
                from .scalar_polynomial import scalar_second
                dx,dc,dy=scalar_second(x,coeff,grad,indices,plan,hx,hc,needs)
                return dx,dc,dy,None,None,None
            dx=dc=dy=None
            if hx is not None:
                hx=hx.contiguous()
                if needs[0]:
                    q=torch.bmm(grad,coeff.transpose(1,2))
                    dx=_adjoint_raw(x,q,plan,indices.shape[0],hx)
                    del q
                if needs[1] or needs[2]:
                    tangent=_value_raw(x,indices,hx)
                    if needs[1]:dc=torch.bmm(tangent.transpose(1,2),grad)
                    if needs[2]:dy=torch.bmm(tangent,coeff)
            if hc is not None:
                if needs[0]:
                    q=torch.bmm(grad,hc.transpose(1,2))
                    partial=_adjoint_raw(x,q,plan,indices.shape[0])
                    dx=partial if dx is None else dx+partial
                    del q
                if needs[2]:
                    phi=_value_raw(x,indices)
                    partial=torch.bmm(phi,hc)
                    dy=partial if dy is None else dy+partial
        return dx,dc,dy,None,None,None


class _PolynomialGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,coeff,indices,plan):
        ctx.save_for_backward(x,coeff,indices,plan)
        if coeff.shape[-1]==1:
            from .scalar_polynomial import scalar_forward
            return scalar_forward(x,coeff,indices)
        return torch.bmm(_value_raw(x,indices),coeff)

    @staticmethod
    def backward(ctx,grad):
        x,coeff,indices,plan=ctx.saved_tensors;needed=required_gradients((x,coeff),ctx.needs_input_grad[:2])
        grad=grad.contiguous()
        if torch.is_grad_enabled():
            dx,dc=_GemmAdjoint.apply(x,coeff,grad,indices,plan,needed)
        else:
            dx,dc=_first(x,coeff,grad,indices,plan,needed)
        return dx,dc,None,None


def polynomial_gemm(x,coeff,indices,plan):
    if triton is not None and x.is_cuda and x.dtype==coeff.dtype==torch.float32:
        return _PolynomialGemm.apply(x,coeff,indices,plan)
    return torch.bmm(monomials(x,indices,plan),coeff)
