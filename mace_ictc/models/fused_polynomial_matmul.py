"""Experimental IEEE FP32 fused polynomial multiply; not the selected route.

Whole-model v28/v29 regress. Kept for diagnosis; polynomial_gemm uses cuBLAS.
"""
import torch
from .monomial_operator import triton
if triton is not None:
    import triton.language as tl

    @triton.jit
    def _polynomial_matmul(X,A,H,J,Y,N:tl.constexpr,C:tl.constexpr,I:tl.constexpr,T:tl.constexpr,M:tl.constexpr,
                           X0:tl.constexpr,X1:tl.constexpr,X2:tl.constexpr,ORDER:tl.constexpr,JVP:tl.constexpr,
                           BN:tl.constexpr,BK:tl.constexpr,BM:tl.constexpr):
        c=tl.program_id(1);n=tl.program_id(0).to(tl.int64)*BN+tl.arange(0,BN)
        k=tl.arange(0,BK);m=tl.arange(0,BM)
        acc=tl.full((BN,BM),0,tl.float32)
        for start in range(0,T,BK):
            t=start+k;valid=(n[:,None]<N)&(t[None,:]<T)
            a=tl.load(J+t,t<T,0);xa=tl.load(X+n[:,None]*X0+c*X1+a[None,:]*X2,valid,0.)
            if JVP:ha=tl.load(H+(n[:,None]*C+c)*I+a[None,:],valid,0.)
            if ORDER==1:phi=ha if JVP else xa
            else:
                b=tl.load(J+T+t,t<T,0);xb=tl.load(X+n[:,None]*X0+c*X1+b[None,:]*X2,valid,0.)
                if JVP:hb=tl.load(H+(n[:,None]*C+c)*I+b[None,:],valid,0.)
                if ORDER==2:phi=ha*xb+xa*hb if JVP else xa*xb
                else:
                    d=tl.load(J+2*T+t,t<T,0);xd=tl.load(X+n[:,None]*X0+c*X1+d[None,:]*X2,valid,0.)
                    if JVP:
                        hd=tl.load(H+(n[:,None]*C+c)*I+d[None,:],valid,0.)
                        phi=(ha*xb+xa*hb)*xd+(xa*xb)*hd
                    else:phi=(xa*xb)*xd
            coefficient=tl.load(A+(c*T+t[:,None])*M+m[None,:],(t[:,None]<T)&(m[None,:]<M),0.)
            acc=tl.dot(phi,coefficient,acc,input_precision="ieee")
        tl.store(Y+(c*N+n[:,None])*M+m[None,:],acc,(n[:,None]<N)&(m[None,:]<M))

    @triton.jit
    def _polynomial_dc(X,G,H,J,O,N:tl.constexpr,C:tl.constexpr,I:tl.constexpr,T:tl.constexpr,M:tl.constexpr,
                       X0:tl.constexpr,X1:tl.constexpr,X2:tl.constexpr,ORDER:tl.constexpr,JVP:tl.constexpr,
                       NS:tl.constexpr,BN:tl.constexpr,BT:tl.constexpr,BM:tl.constexpr):
        c=tl.program_id(1);split=tl.program_id(2)
        t=tl.program_id(0).to(tl.int64)*BT+tl.arange(0,BT);m=tl.arange(0,BM)
        ni=tl.arange(0,BN);acc=tl.full((BT,BM),0,tl.float32)
        a=tl.load(J+t,t<T,0)
        if ORDER>1:b=tl.load(J+T+t,t<T,0)
        if ORDER>2:d=tl.load(J+2*T+t,t<T,0)
        for offset in range(0,256,BN):
            n=split*256+offset+ni;valid=(n[:,None]<N)&(t[None,:]<T)
            xa=tl.load(X+n[:,None]*X0+c*X1+a[None,:]*X2,valid,0.)
            if JVP:ha=tl.load(H+(n[:,None]*C+c)*I+a[None,:],valid,0.)
            if ORDER==1:phi=ha if JVP else xa
            else:
                xb=tl.load(X+n[:,None]*X0+c*X1+b[None,:]*X2,valid,0.)
                if JVP:hb=tl.load(H+(n[:,None]*C+c)*I+b[None,:],valid,0.)
                if ORDER==2:phi=ha*xb+xa*hb if JVP else xa*xb
                else:
                    xd=tl.load(X+n[:,None]*X0+c*X1+d[None,:]*X2,valid,0.)
                    if JVP:
                        hd=tl.load(H+(n[:,None]*C+c)*I+d[None,:],valid,0.)
                        phi=(ha*xb+xa*hb)*xd+(xa*xb)*hd
                    else:phi=(xa*xb)*xd
            grad=tl.load(G+(c*N+n[:,None])*M+m[None,:],(n[:,None]<N)&(m[None,:]<M),0.)
            acc=tl.dot(tl.trans(phi),grad,acc,input_precision="ieee")
        tl.store(O+((c*T+t[:,None])*M+m[None,:])*NS+split,acc,(t[:,None]<T)&(m[None,:]<M))


def polynomial_matmul(x,coeff,indices,h=None):
    n,c,i=x.shape;t,m=coeff.shape[1:]
    out=x.new_empty(c,n,m)
    if out.numel():
        _polynomial_matmul[(triton.cdiv(n,32),c)](x,coeff,x if h is None else h.contiguous(),indices,out,
            n,c,i,t,m,*x.stride(),indices.shape[0],h is not None,32,32,max(16,triton.next_power_of_2(m)),
            num_warps=4,enable_fp_fusion=True)
    return out


def polynomial_weight_gradient(x,grad,indices,h=None):
    n,c,i=x.shape;t=indices.shape[1];m=grad.shape[-1];ns=triton.cdiv(n,256)
    partial=x.new_empty(c,t,m,ns)
    if partial.numel():
        _polynomial_dc[(triton.cdiv(t,32),c,ns)](x,grad,x if h is None else h.contiguous(),indices,partial,
            n,c,i,t,m,*x.stride(),indices.shape[0],h is not None,ns,32,32,max(16,triton.next_power_of_2(m)),
            num_warps=4,enable_fp_fusion=True)
    return partial.sum(-1)
