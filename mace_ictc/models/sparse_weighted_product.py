"""Sparse fixed three-leg maps with independent first and mixed adjoints.

Entries are (left feature, right feature, channel weight, output feature,
constant). The same primitive evaluates CG prefixes and their weighted final
couplings. No node-sized path-weight tensor is formed.
"""
import torch
from .monomial_operator import triton
from .adjoint_requirements import required_gradients

if triton is not None:
    import triton.language as tl

    @triton.jit
    def _map(L,R,W,G,IDS,COEF,Y,N:tl.constexpr,C:tl.constexpr,
             A:tl.constexpr,B:tl.constexpr,P:tl.constexpr,M:tl.constexpr,
             O:tl.constexpr,K:tl.constexpr,MODE:tl.constexpr,
             BO:tl.constexpr,BK:tl.constexpr):
        row=tl.program_id(0).to(tl.int64)*BO+tl.arange(0,BO)
        f=row%O;c=(row//O)%C;n=row//(C*O);valid=row<N*C*O
        k=tl.arange(0,BK);active=valid[:,None]&(k[None,:]<K)
        a=tl.load(IDS+f[:,None]*K+k[None,:],active,0)
        b=tl.load(IDS+O*K+f[:,None]*K+k[None,:],active,0)
        p=tl.load(IDS+2*O*K+f[:,None]*K+k[None,:],active,0)
        q=tl.load(IDS+3*O*K+f[:,None]*K+k[None,:],active,0)
        coefficient=tl.load(COEF+f[:,None]*K+k[None,:],active,0.)
        if MODE<3:w=tl.load(W+c[:,None]*P+p,active,0.)
        if MODE==0:
            l=tl.load(L+(n[:,None]*C+c[:,None])*A+a,active,0.)
            r=tl.load(R+(n[:,None]*C+c[:,None])*B+b,active,0.)
            value=(l*r)*(coefficient*w)
        elif MODE==1:
            r=tl.load(R+(n[:,None]*C+c[:,None])*B+b,active,0.)
            g=tl.load(G+(n[:,None]*C+c[:,None])*M+q,active,0.)
            value=(r*g)*(coefficient*w)
        elif MODE==2:
            l=tl.load(L+(n[:,None]*C+c[:,None])*A+a,active,0.)
            g=tl.load(G+(n[:,None]*C+c[:,None])*M+q,active,0.)
            value=(l*g)*(coefficient*w)
        else:
            l=tl.load(L+(n[:,None]*C+c[:,None])*A+a,active,0.)
            r=tl.load(R+(n[:,None]*C+c[:,None])*B+b,active,0.)
            g=tl.load(G+(n[:,None]*C+c[:,None])*M+q,active,0.)
            value=(l*r)*(coefficient*g)
        tl.store(Y+row,tl.sum(value,1),valid)

    @triton.jit
    def _weight_adjoint(L,R,G,IDS,COEF,Y,N:tl.constexpr,C:tl.constexpr,
             A:tl.constexpr,B:tl.constexpr,P:tl.constexpr,M:tl.constexpr,
             K:tl.constexpr,NB:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
        row=tl.program_id(0);c=row//P;p=row%P
        n=tl.program_id(1).to(tl.int64)*BN+tl.arange(0,BN)
        k=tl.arange(0,BK);active=(n[:,None]<N)&(k[None,:]<K)
        a=tl.load(IDS+p*K+k,k<K,0)
        b=tl.load(IDS+P*K+p*K+k,k<K,0)
        q=tl.load(IDS+3*P*K+p*K+k,k<K,0)
        coefficient=tl.load(COEF+p*K+k,k<K,0.)
        l=tl.load(L+(n[:,None]*C+c)*A+a[None,:],active,0.)
        r=tl.load(R+(n[:,None]*C+c)*B+b[None,:],active,0.)
        g=tl.load(G+(n[:,None]*C+c)*M+q[None,:],active,0.)
        value=tl.sum(tl.sum((l*r)*(g*coefficient[None,:]),1),0)
        tl.store(Y+row*NB+tl.program_id(1),value)


def sparse_product_plan(entries, sizes, device, dtype):
    """Build all unique-owner reductions from immutable exact-nonzero entries."""
    plans=[]
    for leg,size in zip((3,0,1,2),(sizes[3],sizes[0],sizes[1],sizes[2])):
        rows=[[] for _ in range(size)]
        for entry in entries:
            rows[entry[leg]].append(entry)
        k=max(1,max(map(len,rows)))
        padded=[row+[(0,0,0,0,0.)]*(k-len(row)) for row in rows]
        ids=torch.tensor([[entry[:4] for entry in row] for row in padded],
                         device=device,dtype=torch.int32).permute(2,0,1).contiguous()
        coefficients=torch.tensor([[entry[4] for entry in row] for row in padded],
                                  device=device,dtype=dtype)
        plans.append((ids,coefficients))
    return {"sizes":sizes,"reductions":plans,"entries":tuple(entries)}


def _native(l,r,w,plan):
    result=l.new_zeros(*l.shape[:2],plan["sizes"][3])
    # Used for small CPU validation and higher derivatives only.
    for a,b,p,q,value in plan["entries"]:
        result[:,:,q]=result[:,:,q]+(l[:,:,a]*r[:,:,b])*(w[:,p]*value)
    return result


def _raw(l,r,w,plan,mode=0,g=None):
    n,c,_=l.shape;a,b,p,m=plan["sizes"]
    ids,coeff=plan["reductions"][mode]
    bk=triton.next_power_of_2(ids.shape[-1])
    adaptive=plan.get("adaptive_tiling",False)
    bo=max(1,min(16,2048//bk)) if adaptive else 16
    warps=8 if adaptive and bk>2048 else 4
    if mode==3:
        if plan.get("weight_node_reduction",False):
            out=l.new_empty(n,c,p)
            if out.numel():
                _map[(triton.cdiv(out.numel(),bo),)](l,r,w,g,ids,coeff,out,
                    n,c,a,b,p,m,p,ids.shape[-1],3,bo,bk,num_warps=warps)
            return out.sum(0)
        nb=triton.cdiv(n,32);out=l.new_empty(c,p,nb)
        if out.numel():
            _weight_adjoint[(c*p,nb)](l,r,g,ids,coeff,out,n,c,a,b,p,m,
                ids.shape[-1],nb,32,triton.next_power_of_2(ids.shape[-1]),num_warps=4)
        return out.sum(-1)
    o=(m,a,b)[mode];out=l.new_empty(n,c,o)
    if out.numel():
        _map[(triton.cdiv(out.numel(),bo),)](l,r,w,l if g is None else g,ids,coeff,out,
            n,c,a,b,p,m,o,ids.shape[-1],mode,bo,bk,num_warps=warps)
    return out


class _Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx,l,r,w,g,plan,needed):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(l,r,w,g);ctx.plan=plan
        return tuple(_raw(l,r,w,plan,i+1,g) if flag else None
                     for i,flag in enumerate(needed))

    @staticmethod
    def backward(ctx,hl,hr,hw):
        l,r,w,g=ctx.saved_tensors;plan=ctx.plan
        needed=required_gradients((l,r,w,g),ctx.needs_input_grad[:4])
        if torch.is_grad_enabled():
            # Clones isolate partial derivatives from shared ancestors.
            with torch.enable_grad():
                operands=[v.clone() for v in (l,r,w,g)]
                active=[i for i,v in enumerate(operands) if v.requires_grad]
                output=_native(*operands[:3],plan)
                first_active=[i for i in active if i<3]
                derivatives=torch.autograd.grad(output,[operands[i] for i in first_active],operands[3],
                            create_graph=True,allow_unused=True)
                first=[None]*3
                for i,value in zip(first_active,derivatives):first[i]=value
                terms=[(d*h).sum() for d,h in zip(first,(hl,hr,hw)) if d is not None and h is not None]
                values=torch.autograd.grad(sum(terms),[operands[i] for i in active],
                            create_graph=True,allow_unused=True) if terms else [None]*len(active)
                result=[None]*4
                for i,value in zip(active,values):result[i]=value if needed[i] else None
            return *result,None,None
        dl=dr=dw=dg=None
        def add(left,right):return right if left is None else left+right
        if hl is not None:
            hl=hl.contiguous()
            if needed[1]:dr=add(dr,_raw(hl,r,w,plan,2,g))
            if needed[2]:dw=add(dw,_raw(hl,r,w,plan,3,g))
            if needed[3]:dg=add(dg,_raw(hl,r,w,plan))
        if hr is not None:
            hr=hr.contiguous()
            if needed[0]:dl=add(dl,_raw(l,hr,w,plan,1,g))
            if needed[2]:dw=add(dw,_raw(l,hr,w,plan,3,g))
            if needed[3]:dg=add(dg,_raw(l,hr,w,plan))
        if hw is not None:
            hw=hw.contiguous()
            if needed[0]:dl=add(dl,_raw(l,r,hw,plan,1,g))
            if needed[1]:dr=add(dr,_raw(l,r,hw,plan,2,g))
            if needed[3]:dg=add(dg,_raw(l,r,hw,plan))
        return dl,dr,dw,dg,None,None


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx,l,r,w,plan):
        ctx.save_for_backward(l,r,w);ctx.plan=plan
        return _raw(l,r,w,plan)

    @staticmethod
    def backward(ctx,g):
        l,r,w=ctx.saved_tensors
        needed=required_gradients((l,r,w),ctx.needs_input_grad[:3]);g=g.contiguous()
        if torch.is_grad_enabled():values=_Adjoint.apply(l,r,w,g,ctx.plan,needed)
        else:values=tuple(_raw(l,r,w,ctx.plan,i+1,g) if flag else None
                         for i,flag in enumerate(needed))
        return *values,None


def sparse_weighted_product(l,r,w,plan):
    if triton is not None and l.is_cuda and l.dtype==torch.float32:
        return _Product.apply(l.contiguous(),r.contiguous(),w.contiguous(),plan)
    return _native(l,r,w,plan)


class _QuadraticAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,g,ones,plan,adjoint_plan):
        ctx.save_for_backward(x,g);ctx.quad_plan=(ones,plan,adjoint_plan)
        return _raw(x,x,ones,adjoint_plan,1,g)

    @staticmethod
    def backward(ctx,h):
        x,g=ctx.saved_tensors;ones,plan,adjoint_plan=ctx.quad_plan
        needed=required_gradients((x,g),ctx.needs_input_grad[:2])
        if torch.is_grad_enabled():
            xx=x.clone();gg=g.clone()
            first=torch.autograd.grad(_native(xx,xx,ones,plan),xx,gg,create_graph=True)[0]
            active=[value for value,need in zip((xx,gg),needed) if need]
            derivatives=iter(torch.autograd.grad(first,active,h,create_graph=True,allow_unused=True))
            dx,dg=(next(derivatives) if need else None for need in needed)
        else:
            h=h.contiguous()
            dx=_raw(h,h,ones,adjoint_plan,1,g) if needed[0] else None
            dg=_raw(h,x,ones,adjoint_plan) if needed[1] else None
        return dx,dg,None,None,None


class _Quadratic(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,ones,plan,adjoint_plan):
        ctx.save_for_backward(x);ctx.quad_plan=(ones,plan,adjoint_plan)
        return _raw(x,x,ones,plan)

    @staticmethod
    def backward(ctx,g):
        x,=ctx.saved_tensors;ones,plan,adjoint_plan=ctx.quad_plan;g=g.contiguous()
        value=(_QuadraticAdjoint.apply(x,g,ones,plan,adjoint_plan) if torch.is_grad_enabled()
               else _raw(x,x,ones,adjoint_plan,1,g))
        return value,None,None,None


def sparse_quadratic(x,ones,plan,adjoint_plan):
    if triton is not None and x.is_cuda and x.dtype==torch.float32:
        return _Quadratic.apply(x.contiguous(),ones,plan,adjoint_plan)
    return _native(x,x,ones,plan)
