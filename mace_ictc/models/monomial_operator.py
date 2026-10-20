"""Independent FP32 monomials with compact first/second derivative kernels."""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _value(X,H,J,Y,N:tl.constexpr,C:tl.constexpr,I:tl.constexpr,T:tl.constexpr,
               ORDER:tl.constexpr,X0:tl.constexpr,X1:tl.constexpr,X2:tl.constexpr,
               JVP:tl.constexpr,BLOCK:tl.constexpr):
        index=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
        valid=index<C*N*T;t=index%T;n=(index//T)%N;c=index//(N*T)
        a=tl.load(J+t,t<T,0)
        xa=tl.load(X+n*X0+c*X1+a*X2,valid,0.)
        if JVP:ha=tl.load(H+(n*C+c)*I+a,valid,0.)
        if ORDER==1:
            result=ha if JVP else xa
        else:
            b=tl.load(J+T+t,t<T,0);xb=tl.load(X+n*X0+c*X1+b*X2,valid,0.)
            if JVP:hb=tl.load(H+(n*C+c)*I+b,valid,0.)
            if ORDER==2:
                result=ha*xb+xa*hb if JVP else xa*xb
            else:
                d=tl.load(J+2*T+t,t<T,0);xd=tl.load(X+n*X0+c*X1+d*X2,valid,0.)
                if JVP:
                    hd=tl.load(H+(n*C+c)*I+d,valid,0.)
                    result=(ha*xb+xa*hb)*xd+(xa*xb)*hd
                else:result=(xa*xb)*xd
        tl.store(Y+index,result,valid)

    @triton.jit
    def _adjoint(X,G,H,PLAN,Y,N:tl.constexpr,C:tl.constexpr,I:tl.constexpr,T:tl.constexpr,
                 K:tl.constexpr,ORDER:tl.constexpr,X0:tl.constexpr,X1:tl.constexpr,
                 X2:tl.constexpr,HESSIAN:tl.constexpr,CMAJOR:tl.constexpr,BO:tl.constexpr,BK:tl.constexpr):
        row=tl.program_id(0).to(tl.int64)*BO+tl.arange(0,BO)
        if CMAJOR:n=(row//I)%N;c=row//(N*I)
        else:n=row//(C*I);c=(row//I)%C
        i=row%I;valid=row<N*C*I
        k=tl.arange(0,BK);active=valid[:,None]&(k[None,:]<K)
        t=tl.load(PLAN+i[:,None]*K+k[None,:],active,-1)
        active=active&(t>=0)
        a=tl.load(PLAN+I*K+i[:,None]*K+k[None,:],active,0)
        b=tl.load(PLAN+2*I*K+i[:,None]*K+k[None,:],active,0)
        multiplicity=tl.load(PLAN+3*I*K+i[:,None]*K+k[None,:],active,0)
        g=tl.load(G+(c[:,None]*N+n[:,None])*T+t,active,0.)
        if ORDER==1:
            factor=tl.full((BO,BK),0 if HESSIAN else 1,tl.float32)
        else:
            xa=tl.load(X+n[:,None]*X0+c[:,None]*X1+a*X2,active,0.)
            if HESSIAN:ha=tl.load(H+(n[:,None]*C+c[:,None])*I+a,active,0.)
            if ORDER==2:factor=ha if HESSIAN else xa
            else:
                xb=tl.load(X+n[:,None]*X0+c[:,None]*X1+b*X2,active,0.)
                if HESSIAN:
                    hb=tl.load(H+(n[:,None]*C+c[:,None])*I+b,active,0.)
                    factor=ha*xb+xa*hb
                else:factor=xa*xb
        result=tl.sum((g*multiplicity)*factor,axis=1)
        tl.store(Y+(n*C+c)*I+i,result,valid)


def monomial_plan(tuples, size, device):
    rows=[]
    for i in range(size):
        entries=[]
        for t, indices in enumerate(tuples):
            count=indices.count(i)
            if count:
                others=list(indices);others.remove(i);others += [0]*(2-len(others))
                entries.append((t,others[0],others[1],count))
        rows.append(entries)
    k=max(map(len,rows))
    rows=[entries+[(-1,0,0,0)]*(k-len(entries)) for entries in rows]
    return torch.tensor(rows,device=device,dtype=torch.int32).permute(2,0,1).contiguous()


def _native(x, indices):
    out=x.index_select(-1,indices[0].long())
    for leg in range(1,indices.shape[0]):
        out=out*x.index_select(-1,indices[leg].long())
    return out.transpose(0,1).contiguous()


def _native_jvp(x,indices,h):
    parts=[]
    for leg in range(indices.shape[0]):
        value=h.index_select(-1,indices[leg].long())
        for other in range(indices.shape[0]):
            if other!=leg:value=value*x.index_select(-1,indices[other].long())
        parts.append(value)
    return sum(parts).transpose(0,1).contiguous()


def _native_hessian(x,grad,plan,order,h):
    n,c,i=x.shape;k=plan.shape[-1]
    t,a,b,count=plan.long()
    values=grad.transpose(0,1).index_select(-1,t.clamp_min(0).reshape(-1)).reshape(n,c,i,k)
    if order==1:return x*0
    ha=h.index_select(-1,a.reshape(-1)).reshape(n,c,i,k)
    if order==2:factor=ha
    else:
        xa=x.index_select(-1,a.reshape(-1)).reshape(n,c,i,k)
        xb=x.index_select(-1,b.reshape(-1)).reshape(n,c,i,k)
        hb=h.index_select(-1,b.reshape(-1)).reshape(n,c,i,k)
        factor=ha*xb+xa*hb
    return ((values*count)*factor).sum(-1)


def _value_raw(x,indices,h=None):
    n,c,i=x.shape;t=indices.shape[1]
    out=x.new_empty(c,n,t)
    if out.numel():
        _value[(triton.cdiv(out.numel(),256),)](x,x if h is None else h,indices,out,n,c,i,t,
            indices.shape[0],*x.stride(),h is not None,256,num_warps=4,enable_fp_fusion=True)
    return out


def _adjoint_raw(x,grad,plan,order,h=None,*,rows=None,warps=4,cmajor=False):
    n,c,i=x.shape
    if not plan.shape[-1]:return x.new_zeros(n,c,i)
    if rows is None:rows=8 if plan.shape[-1]>128 else 16
    out=x.new_empty(n,c,i)
    if out.numel():
        _adjoint[(triton.cdiv(out.numel(),rows),)](x,grad.contiguous(),x if h is None else h,plan,out,
            n,c,i,grad.shape[-1],plan.shape[-1],order,
            *x.stride(),h is not None,cmajor,rows,triton.next_power_of_2(plan.shape[-1]),num_warps=warps,enable_fp_fusion=True)
    return out


class _MonomialAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,grad,indices,plan):
        ctx.save_for_backward(x,grad,indices,plan)
        return _adjoint_raw(x,grad,plan,indices.shape[0])

    @staticmethod
    def backward(ctx,h):
        x,grad,indices,plan=ctx.saved_tensors
        if torch.is_grad_enabled():
            dx=_native_hessian(x,grad,plan,indices.shape[0],h) if ctx.needs_input_grad[0] else None
            dg=_native_jvp(x,indices,h) if ctx.needs_input_grad[1] else None
        else:
            h=h.contiguous()
            dx=_adjoint_raw(x,grad,plan,indices.shape[0],h) if ctx.needs_input_grad[0] else None
            dg=_value_raw(x,indices,h) if ctx.needs_input_grad[1] else None
        return dx,dg,None,None


class _Monomials(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,indices,plan):
        ctx.save_for_backward(x,indices,plan)
        return _value_raw(x,indices)

    @staticmethod
    def backward(ctx,grad):
        x,indices,plan=ctx.saved_tensors
        dx=(_MonomialAdjoint.apply(x,grad,indices,plan) if torch.is_grad_enabled()
            else _adjoint_raw(x,grad,plan,indices.shape[0]))
        return dx,None,None


def monomials(x,indices,plan):
    if triton is not None and x.is_cuda and x.dtype==torch.float32:
        return _Monomials.apply(x,indices,plan)
    return _native(x,indices)
