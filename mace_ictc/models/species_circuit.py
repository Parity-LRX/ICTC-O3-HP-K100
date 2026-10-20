# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Live species-indexed CG circuits without host-sized node selections.

The original CG circuit and learned coefficient bank are unchanged. Each node
loads its own species row. Weight adjoints use bounded partial reductions; no
atomic floating-point accumulation or cached feature/species partition is used.
"""
import torch
from .adjoint_requirements import required_gradients
from .factorized_circuit import _raw, _reference
from .monomial_operator import triton

if triton is not None:
    import triton.language as tl

    @triton.jit
    def _reduce_bank(V,IDX,O,N,C:tl.constexpr,P:tl.constexpr,S:tl.constexpr,
                     CHUNKS:tl.constexpr,BN:tl.constexpr,BC:tl.constexpr):
        chunk=tl.program_id(0).to(tl.int64)
        pc=tl.program_id(1).to(tl.int64)*BC+tl.arange(0,BC)
        n=chunk*BN+tl.arange(0,BN)
        species=tl.load(IDX+n,n<N,0)
        value=tl.load(V+(pc[None,:]//C*N+n[:,None])*C+pc[None,:]%C,
                      (n[:,None]<N)&(pc[None,:]<P*C),0.)
        for s in tl.static_range(S):
            total=tl.sum(tl.where(species[:,None]==s,value,0.),axis=0)
            tl.store(O+(s*CHUNKS+chunk)*(P*C)+pc,total,pc<P*C)


def reduce_bank(values,index,species):
    p,n,c=values.shape
    chunks=triton.cdiv(n,256)
    partial=values.new_empty(species,chunks,p,c)
    if partial.numel():
        _reduce_bank[(chunks,triton.cdiv(p*c,16))](values,index,partial,n,c,p,species,chunks,256,16,num_warps=4)
    return partial.sum(1).transpose(1,2).contiguous()


def reference(x,bank,index,plan):
    result=None
    for species in range(bank.shape[0]):
        value=_reference(x,bank[species],plan)
        value=torch.where(index[:,None,None]==species,value,0.)
        result=value if result is None else result+value
    return result


class _BankAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,a,g,index,plan,needed):
        ctx.set_materialize_grads(False);ctx.save_for_backward(x,a,g,index)
        ctx.plan=plan;ctx.needed=needed
        return (_raw(x,a,plan,1,g,idx=index) if needed[0] else None,
                _raw(x,a,plan,2,g,idx=index) if needed[1] else None)

    @staticmethod
    def backward(ctx,hx,ha):
        x,a,g,index=ctx.saved_tensors;plan=ctx.plan
        needed=required_gradients((x,a,g),ctx.needs_input_grad[:3])
        if torch.is_grad_enabled():
            local=[v.clone() for v in (x,a,g)]
            active_first=[v for v,need in zip(local[:2],ctx.needed) if need]
            first=torch.autograd.grad(reference(local[0],local[1],index,plan),active_first,local[2],create_graph=True)
            incoming=[v for v,need in zip((hx,ha),ctx.needed) if need]
            terms=[(d*h).sum() for d,h in zip(first,incoming) if h is not None]
            if not terms:return None,None,None,None,None,None
            active=[v for v,need in zip(local,needed) if need]
            derivatives=iter(torch.autograd.grad(sum(terms),active,create_graph=True,allow_unused=True))
            result=[next(derivatives) if need else None for need in needed]
            return *result,None,None,None
        dx=_raw(x,a,plan,1,g,hx,ha,idx=index) if needed[0] and (hx is not None or ha is not None) else None
        da=_raw(x,a,plan,2,g,hx,idx=index) if needed[1] and hx is not None else None
        dg=_raw(x,a,plan,3,None,hx,ha,idx=index) if needed[2] and (hx is not None or ha is not None) else None
        return dx,da,dg,None,None,None


class _BankCircuit(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,a,index,plan):
        ctx.save_for_backward(x,a,index);ctx.plan=plan
        return _raw(x,a,plan,idx=index)

    @staticmethod
    def backward(ctx,g):
        x,a,index=ctx.saved_tensors;plan=ctx.plan;g=g.contiguous()
        needed=required_gradients((x,a),ctx.needs_input_grad[:2])
        if torch.is_grad_enabled():values=_BankAdjoint.apply(x,a,g,index,plan,needed)
        else:values=(_raw(x,a,plan,1,g,idx=index) if needed[0] else None,
                     _raw(x,a,plan,2,g,idx=index) if needed[1] else None)
        return *values,None,None


def species_circuit(x,bank,index,plan):
    if index.shape!=(x.shape[0],) or index.device!=x.device or index.dtype!=torch.int64:
        raise ValueError('Species indices must be int64 [N] on the feature device')
    torch._assert_async(((index>=0)&(index<bank.shape[0])).all(),'Species index outside the coefficient bank')
    if triton is not None and x.is_cuda and x.dtype==torch.float32:
        return _BankCircuit.apply(x,bank,index.contiguous(),plan)
    return reference(x,bank,index,plan)
