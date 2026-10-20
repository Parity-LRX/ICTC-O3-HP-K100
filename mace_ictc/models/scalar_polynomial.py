"""Scalar polynomial and adjoints without an expanded node/term cotangent.

For one output, the cotangent is one value per node/channel. Fuse its
broadcast into the derivative reduction instead of storing a dense N*C*T map.
"""
import torch
from .monomial_operator import triton
if triton is not None:
    import triton.language as tl

    @triton.jit
    def _scalar_value(X,A,H,J,Y,N:tl.constexpr,C:tl.constexpr,I:tl.constexpr,T:tl.constexpr,
               ORDER:tl.constexpr,JVP:tl.constexpr,BO:tl.constexpr,BK:tl.constexpr):
        row=tl.program_id(0).to(tl.int64)*BO+tl.arange(0,BO)
        c=row//N;n=row%N;valid=row<C*N
        t=tl.arange(0,BK);active=valid[:,None]&(t[None,:]<T)
        a=tl.load(J+t,t<T,0);xa=tl.load(X+(c[:,None]*N+n[:,None])*I+a[None,:],active,0.)
        if JVP:ha=tl.load(H+(c[:,None]*N+n[:,None])*I+a[None,:],active,0.)
        if ORDER==1:value=ha if JVP else xa
        else:
            b=tl.load(J+T+t,t<T,0);xb=tl.load(X+(c[:,None]*N+n[:,None])*I+b[None,:],active,0.)
            if JVP:hb=tl.load(H+(c[:,None]*N+n[:,None])*I+b[None,:],active,0.)
            if ORDER==2:value=ha*xb+xa*hb if JVP else xa*xb
            else:
                d=tl.load(J+2*T+t,t<T,0);xd=tl.load(X+(c[:,None]*N+n[:,None])*I+d[None,:],active,0.)
                if JVP:
                    hd=tl.load(H+(c[:,None]*N+n[:,None])*I+d[None,:],active,0.)
                    value=(ha*xb+xa*hb)*xd+(xa*xb)*hd
                else:value=(xa*xb)*xd
        coefficient=tl.load(A+c[:,None]*T+t[None,:],active,0.)
        tl.store(Y+row,tl.sum(value*coefficient,1),valid)

    @triton.jit
    def _scalar_dx(X,A,G,H,PLAN,Y,N:tl.constexpr,C:tl.constexpr,I:tl.constexpr,T:tl.constexpr,
            K:tl.constexpr,ORDER:tl.constexpr,HESSIAN:tl.constexpr,BO:tl.constexpr,BK:tl.constexpr):
        row=tl.program_id(0).to(tl.int64)*BO+tl.arange(0,BO)
        c=row//(N*I);n=(row//I)%N;i=row%I;valid=row<C*N*I
        k=tl.arange(0,BK);active=valid[:,None]&(k[None,:]<K)
        t=tl.load(PLAN+i[:,None]*K+k[None,:],active,-1);active=active&(t>=0)
        a=tl.load(PLAN+I*K+i[:,None]*K+k[None,:],active,0)
        b=tl.load(PLAN+2*I*K+i[:,None]*K+k[None,:],active,0)
        count=tl.load(PLAN+3*I*K+i[:,None]*K+k[None,:],active,0)
        coefficient=tl.load(A+c[:,None]*T+t,active,0.)
        if ORDER==1:factor=tl.full((BO,BK),0 if HESSIAN else 1,tl.float32)
        else:
            xa=tl.load(X+(c[:,None]*N+n[:,None])*I+a,active,0.)
            if HESSIAN:ha=tl.load(H+(c[:,None]*N+n[:,None])*I+a,active,0.)
            if ORDER==2:factor=ha if HESSIAN else xa
            else:
                xb=tl.load(X+(c[:,None]*N+n[:,None])*I+b,active,0.)
                if HESSIAN:
                    hb=tl.load(H+(c[:,None]*N+n[:,None])*I+b,active,0.)
                    factor=ha*xb+xa*hb
                else:factor=xa*xb
        g=tl.load(G+c*N+n,valid,0.)
        result=tl.sum((coefficient*count)*factor,1)*g
        tl.store(Y+(n*C+c)*I+i,result,valid)

    @triton.jit
    def _scalar_dc(X,G,H,J,PARTIAL,N:tl.constexpr,C:tl.constexpr,I:tl.constexpr,T:tl.constexpr,
            NB:tl.constexpr,ORDER:tl.constexpr,JVP:tl.constexpr,BO:tl.constexpr,BN:tl.constexpr):
        row=tl.program_id(0).to(tl.int64)*BO+tl.arange(0,BO)
        c=row//T;t=row%T;valid=row<C*T
        n=tl.program_id(1).to(tl.int64)*BN+tl.arange(0,BN)
        active=valid[:,None]&(n[None,:]<N)
        a=tl.load(J+t,valid,0)
        xa=tl.load(X+(c[:,None]*N+n[None,:])*I+a[:,None],active,0.)
        if JVP:ha=tl.load(H+(c[:,None]*N+n[None,:])*I+a[:,None],active,0.)
        if ORDER==1:value=ha if JVP else xa
        else:
            b=tl.load(J+T+t,valid,0);xb=tl.load(X+(c[:,None]*N+n[None,:])*I+b[:,None],active,0.)
            if JVP:hb=tl.load(H+(c[:,None]*N+n[None,:])*I+b[:,None],active,0.)
            if ORDER==2:value=ha*xb+xa*hb if JVP else xa*xb
            else:
                d=tl.load(J+2*T+t,valid,0);xd=tl.load(X+(c[:,None]*N+n[None,:])*I+d[:,None],active,0.)
                if JVP:
                    hd=tl.load(H+(c[:,None]*N+n[None,:])*I+d[:,None],active,0.)
                    value=(ha*xb+xa*hb)*xd+(xa*xb)*hd
                else:value=(xa*xb)*xd
        g=tl.load(G+c[:,None]*N+n[None,:],active,0.)
        result=tl.sum(value*g,1)
        tl.store(PARTIAL+row*NB+tl.program_id(1),result,valid)


def _value_raw(xc,coeff,indices,hc=None):
    c,n,i=xc.shape;t=indices.shape[1]
    out=xc.new_empty(c,n,1)
    if out.numel():
        _scalar_value[(triton.cdiv(c*n,4),)](xc,coeff,xc if hc is None else hc,indices,out,
            n,c,i,t,indices.shape[0],hc is not None,4,triton.next_power_of_2(max(t,1)),num_warps=4,enable_fp_fusion=True)
    return out


def _dx_raw(xc,coeff,grad,indices,plan,hc=None):
    c,n,i=xc.shape;t=indices.shape[1];k=plan.shape[-1]
    out=xc.new_empty(n,c,i)
    if out.numel():
        _scalar_dx[(triton.cdiv(n*c*i,16),)](xc,coeff,grad,xc if hc is None else hc,plan,out,
            n,c,i,t,k,indices.shape[0],hc is not None,16,triton.next_power_of_2(max(k,1)),num_warps=4,enable_fp_fusion=True)
    return out


def _dc_raw(xc,grad,indices,hc=None):
    c,n,i=xc.shape;t=indices.shape[1];nb=triton.cdiv(n,256)
    partial=xc.new_empty(c,t,nb)
    if partial.numel():
        _scalar_dc[(triton.cdiv(c*t,4),nb)](xc,grad,xc if hc is None else hc,indices,partial,
            n,c,i,t,nb,indices.shape[0],hc is not None,4,256,num_warps=4,enable_fp_fusion=True)
    return partial.sum(-1,keepdim=True)


def scalar_forward(x,coeff,indices):
    return _value_raw(x.transpose(0,1).contiguous(),coeff,indices)


def scalar_first(x,coeff,grad,indices,plan,needed):
    xc=x.transpose(0,1).contiguous()
    dx=_dx_raw(xc,coeff,grad,indices,plan) if needed[0] else None
    dc=_dc_raw(xc,grad,indices) if needed[1] else None
    return dx,dc


def scalar_second(x,coeff,grad,indices,plan,hx,hc,needed):
    xc=x.transpose(0,1).contiguous();dx=dc=dy=None
    if hx is not None:
        hxc=hx.transpose(0,1).contiguous()
        if needed[0]:dx=_dx_raw(xc,coeff,grad,indices,plan,hxc)
        if needed[1]:dc=_dc_raw(xc,grad,indices,hxc)
        if needed[2]:dy=_value_raw(xc,coeff,indices,hxc)
    if hc is not None:
        hc=hc.contiguous()
        if needed[0]:
            partial=_dx_raw(xc,hc,grad,indices,plan)
            dx=partial if dx is None else dx+partial
        if needed[2]:
            partial=_value_raw(xc,hc,indices)
            dy=partial if dy is None else dy+partial
    return dx,dc,dy
