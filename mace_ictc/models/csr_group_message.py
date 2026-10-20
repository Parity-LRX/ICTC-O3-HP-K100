"""Channel-contiguous unique-owner edge reductions with live CSR topology."""
import torch
from .monomial_operator import triton

if triton is not None:
    import triton.language as tl

    @triton.jit
    def _destination(X,B,W,G,HX,HB,HW,HG,S,M,RP,PERM,PM,O,N,E,
            C:tl.constexpr,XDIM:tl.constexpr,BDIM:tl.constexpr,ODIM:tl.constexpr,P:tl.constexpr,
            NP:tl.constexpr,I:tl.constexpr,OM:tl.constexpr,SECOND:tl.constexpr,
            HASX:tl.constexpr,HASB:tl.constexpr,HASW:tl.constexpr,HASG:tl.constexpr,
            MASK:tl.constexpr,BLOCK:tl.constexpr):
        index=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
        valid=index<N*NP*OM*C;node=index//(NP*OM*C)
        pi=(index//(OM*C))%NP;m=(index//C)%OM;c=index%C
        p=tl.load(PM+pi);xb=tl.load(PM+NP+pi);bb=tl.load(PM+2*NP+pi)
        bs=tl.load(PM+3*NP+pi);ob=tl.load(PM+4*NP+pi)
        begin=tl.load(RP+node,valid,0);end=tl.load(RP+node+1,valid,0)
        w=tl.load(W+p*C+c,valid,0.)
        if SECOND and HASW:hw=tl.load(HW+p*C+c,valid,0.)
        result=tl.full((BLOCK,),0,tl.float32)
        count=tl.max(end-begin,0)
        for offset in range(count):
            active=valid&(begin+offset<end)
            e=tl.load(PERM+begin+offset,active,0)
            src=tl.load(S+e,active,0);y=tl.full((BLOCK,),0,tl.float32)
            hy=tl.full((BLOCK,),0,tl.float32)
            for i in tl.static_range(I):
                x=tl.load(X+src*XDIM+xb+i*C+c,active,0.)
                b=tl.load(B+e*BDIM+bb+i*bs+m,active,0.)
                y=y+x*b
                if SECOND and HASX:hy=hy+tl.load(HX+src*XDIM+xb+i*C+c,active,0.)*b
                if SECOND and HASB:hy=hy+x*tl.load(HB+e*BDIM+bb+i*bs+m,active,0.)
            g=tl.load(G+(e*P+p)*C+c,active,0.)
            if SECOND:
                factor=tl.full((BLOCK,),0,tl.float32)
                if HASW:factor=factor+hw*g
                if HASG:factor=factor+w*tl.load(HG+(e*P+p)*C+c,active,0.)
                value=factor*y+(w*g)*hy
            else:value=(y*w)*g
            if MASK:value=value*tl.load(M+e,active,0.)
            result=result+value
        tl.store(O+node*ODIM+ob+m*C+c,result,valid)

    @triton.jit
    def _source(B,W,G,HB,HW,HG,D,M,V,RP,PERM,PM,O,N,E,
            C:tl.constexpr,XDIM:tl.constexpr,BDIM:tl.constexpr,ODIM:tl.constexpr,P:tl.constexpr,
            XBASE:tl.constexpr,I:tl.constexpr,K:tl.constexpr,SECOND:tl.constexpr,
            HASB:tl.constexpr,HASW:tl.constexpr,HASG:tl.constexpr,
            MASK:tl.constexpr,BLOCK:tl.constexpr):
        index=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
        valid=index<N*I*C;node=index//(I*C);i=(index//C)%I;c=index%C
        begin=tl.load(RP+node,valid,0);end=tl.load(RP+node+1,valid,0)
        result=tl.full((BLOCK,),0,tl.float32)
        count=tl.max(end-begin,0)
        for offset in range(count):
            active=valid&(begin+offset<end)
            e=tl.load(PERM+begin+offset,active,0);dst=tl.load(D+e,active,0)
            value=tl.full((BLOCK,),0,tl.float32)
            for k in tl.static_range(K):
                p=tl.load(PM+k);bb=tl.load(PM+K+k);bs=tl.load(PM+2*K+k);ob=tl.load(PM+3*K+k)
                v=tl.load(V+dst*ODIM+ob+c,active,0.)
                if MASK:v=v*tl.load(M+e,active,0.)
                w=tl.load(W+p*C+c,active,0.);g=tl.load(G+(e*P+p)*C+c,active,0.)
                b=tl.load(B+e*BDIM+bb+i*bs,active,0.)
                if SECOND:
                    factor=tl.full((BLOCK,),0,tl.float32)
                    if HASW:factor=factor+tl.load(HW+p*C+c,active,0.)*g
                    if HASG:factor=factor+w*tl.load(HG+(e*P+p)*C+c,active,0.)
                    term=factor*b
                    if HASB:term=term+(w*g)*tl.load(HB+e*BDIM+bb+i*bs,active,0.)
                    value=value+v*term
                else:value=value+((v*g)*w)*b
            result=result+value
        tl.store(O+node*XDIM+XBASE+i*C+c,result,valid)


def _csr(indices,n):
    permutation=torch.argsort(indices,stable=True)
    counts=torch.bincount(indices,minlength=n)
    ptr=torch.cat((counts.new_zeros(1),counts.cumsum(0)))
    return ptr,permutation


def live_csr_plan(plan,dst,n):
    return {**plan,"destination_csr":_csr(dst,n)}


def destination_raw(x,b,w,g,src,mask,plan,incoming=None):
    from .fused_message import _channel_node
    n=x.shape[0];c=plan["c"];e=src.numel();out=x.new_empty(n,plan["odim"])
    xc=_channel_node(x,plan)
    if incoming is None:flags=(False,)*4;hxc,hb,hw,hg=xc,b,w,g
    else:
        flags=tuple(value is not None for value in incoming)
        hx,hb,hw,hg=[value.contiguous() if value is not None else original
                    for value,original in zip(incoming,(x,b,w,g))]
        hxc=_channel_node(hx,plan) if flags[0] else xc
    ptr,permutation=plan["destination_csr"]
    for ni,m,paths in plan["gate_groups"]:
        if n:
            _destination[(triton.cdiv(n*paths.shape[1]*m*c,128),)](xc,b,w,g,hxc,hb,hw,hg,src,mask,
                ptr,permutation,paths,out,n,e,c,plan["xdim"],plan["bdim"],plan["odim"],plan["p"],
                paths.shape[1],ni,m,incoming is not None,*flags,bool(mask.numel()),128,num_warps=4)
    return torch.cat([out[:,plan["ooffset"][l]:plan["ooffset"][l]+c*plan["counts"][l]*(2*l+1)].reshape(
        n,plan["counts"][l],2*l+1,c).transpose(2,3).reshape(n,c*plan["counts"][l]*(2*l+1))
        for l in range(plan["lmax"]+1) if plan["counts"].get(l,0)],dim=1).contiguous()


def source_raw(b,w,g,src,dst,mask,vc,out,plan,incoming=None):
    n=out.shape[0];c=plan["c"];e=src.numel()
    if "source_csr" not in plan:plan["source_csr"]=_csr(src,n)
    ptr,permutation=plan["source_csr"]
    if incoming is None:flags=(False,)*3;hb,hw,hg=b,w,g
    else:
        flags=tuple(value is not None for value in incoming[1:])
        hb,hw,hg=[value.contiguous() if value is not None else original
                  for value,original in zip(incoming[1:],(b,w,g))]
    for xb,ni,paths in plan["dx_channel_groups"]:
        if n:
            _source[(triton.cdiv(n*ni*c,128),)](b,w,g,hb,hw,hg,dst,mask,vc,ptr,permutation,paths,out,
                n,e,c,plan["xdim"],plan["bdim"],plan["odim"],plan["p"],xb,ni,paths.shape[1],
                incoming is not None,*flags,bool(mask.numel()),128,num_warps=4)
