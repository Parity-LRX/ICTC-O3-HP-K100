"""Reduce channel contractions directly into the small harmonic cotangent."""
from .monomial_operator import triton
if triton is not None:
    import triton.language as tl

    @triton.jit
    def _harmonic_term(X,V,W,G,e,src,dst,C:tl.constexpr,XDIM:tl.constexpr,ODIM:tl.constexpr,P:tl.constexpr,
                       PATH:tl.constexpr,XP:tl.constexpr,OB:tl.constexpr,A:tl.constexpr,BC:tl.constexpr,valid):
        c=tl.arange(0,BC)
        x=tl.load(X+src[:,None]*XDIM+XP+c[None,:],valid,0.)
        v=tl.load(V+dst[:,None]*ODIM+OB+c[None,:],valid,0.)
        w=tl.load(W+PATH*C+c,c<C,0.)
        g=tl.load(G+(e[:,None]*P+PATH)*C+c[None,:],valid,0.)
        return x*((v*g)*w[None,:])*A

    @triton.jit
    def _harmonic(X,W,G,S,D,M,V,O,E,C:tl.constexpr,XDIM:tl.constexpr,ODIM:tl.constexpr,
                  P:tl.constexpr,H:tl.constexpr,HIDX:tl.constexpr,PATHS:tl.constexpr,
                  MASK:tl.constexpr,BE:tl.constexpr,BC:tl.constexpr):
        e=tl.program_id(0).to(tl.int64)*BE+tl.arange(0,BE);c=tl.arange(0,BC)
        valid=(e[:,None]<E)&(c[None,:]<C)
        src=tl.load(S+e,e<E,0);dst=tl.load(D+e,e<E,0)
        total=tl.full((BE,BC),0,tl.float32)
        for index in tl.static_range(len(PATHS)):
            total=total+_harmonic_term(X,V,W,G,e,src,dst,C,XDIM,ODIM,P,
                PATHS.value[index][0],PATHS.value[index][1],PATHS.value[index][2],PATHS.value[index][3],BC,valid)
        value=tl.sum(total,axis=1)
        if MASK:value=value*tl.load(M+e,e<E,0.)
        tl.store(O+e*H+HIDX,value,e<E)


def _terms(p,projection):
    old=projection.get("direct_harmonic_terms")
    if old is not None:return old
    result=[]
    for row in projection["static_rows"]:
        entries=[]
        for column,coefficient in row:
            path,xp,_,ob,_=p["bm_static"][column]
            entries.append((path,xp,ob,coefficient))
        result.append(tuple(entries))
    projection["direct_harmonic_terms"]=tuple(result)
    return projection["direct_harmonic_terms"]


def direct_harmonic_raw(x,w,g,src,dst,mask,v,out,p,projection):
    edges=src.numel();be=p.get("harmonic_tile_edges",8);bc=triton.next_power_of_2(p["c"])
    for h,paths in enumerate(_terms(p,projection)):
        _harmonic[(triton.cdiv(edges,be),)](x,w,g,src,dst,mask,v,out,edges,p["c"],p["xdim"],p["odim"],
            p["p"],projection["hdim"],h,paths,bool(mask.numel()),be,bc,num_warps=4,enable_fp_fusion=True)
