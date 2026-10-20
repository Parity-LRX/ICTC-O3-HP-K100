"""Path-specialized FP32 messages and joint feature/geometry/gate adjoints.

Each edge tile loads channel vectors once, reuses the geometry contraction
for both feature and gate gradients, and reduces channels for geometry in the
same kernel. This path uses IEEE FP32 and no cuEq or Tensor Core arithmetic.
"""
from .monomial_operator import triton
if triton is not None:
    import triton.language as tl

    @triton.jit
    def _forward_path(X,B,W,G,S,D,M,O,E,C:tl.constexpr,XDIM:tl.constexpr,BDIM:tl.constexpr,
            ODIM:tl.constexpr,P:tl.constexpr,PATH:tl.constexpr,XB:tl.constexpr,BB:tl.constexpr,
            BS:tl.constexpr,OB:tl.constexpr,I:tl.constexpr,OM:tl.constexpr,
            MASK:tl.constexpr,BE:tl.constexpr,BI:tl.constexpr,BM:tl.constexpr,BC:tl.constexpr):
        e=tl.program_id(0).to(tl.int64)*BE+tl.arange(0,BE)
        i=tl.arange(0,BI);m=tl.arange(0,BM);c=tl.arange(0,BC)
        src=tl.load(S+e,e<E,0);dst=tl.load(D+e,e<E,0)
        x=tl.load(X+src[:,None,None]*XDIM+XB+i[None,:,None]*C+c[None,None,:],
                  (e[:,None,None]<E)&(i[None,:,None]<I)&(c[None,None,:]<C),0.)
        b=tl.load(B+e[:,None,None]*BDIM+BB+i[None,:,None]*BS+m[None,None,:],
                  (e[:,None,None]<E)&(i[None,:,None]<I)&(m[None,None,:]<OM),0.)
        y=tl.sum(x[:,:,None,:]*b[:,:,:,None],axis=1)
        w=tl.load(W+PATH*C+c,c<C,0.)
        g=tl.load(G+(e[:,None]*P+PATH)*C+c[None,:],(e[:,None]<E)&(c[None,:]<C),0.)
        value=(y*w[None,None,:])*g[:,None,:]
        if MASK:value=value*tl.load(M+e,e<E,0.)[:,None,None]
        tl.atomic_add(O+dst[:,None,None]*ODIM+OB+m[None,:,None]*C+c[None,None,:],value,
                      (e[:,None,None]<E)&(m[None,:,None]<OM)&(c[None,None,:]<C),sem="relaxed")

    @triton.jit
    def _adjoint_path(X,B,W,G,S,D,M,V,DX,DB,DG,DWE,E,C:tl.constexpr,XDIM:tl.constexpr,
            BDIM:tl.constexpr,ODIM:tl.constexpr,P:tl.constexpr,PATH:tl.constexpr,
            XB:tl.constexpr,BB:tl.constexpr,BS:tl.constexpr,OB:tl.constexpr,I:tl.constexpr,OM:tl.constexpr,
            NEEDX:tl.constexpr,NEEDB:tl.constexpr,NEEDG:tl.constexpr,NEEDW:tl.constexpr,
            MASK:tl.constexpr,BE:tl.constexpr,BI:tl.constexpr,BM:tl.constexpr,BC:tl.constexpr):
        e=tl.program_id(0).to(tl.int64)*BE+tl.arange(0,BE)
        i=tl.arange(0,BI);m=tl.arange(0,BM);c=tl.arange(0,BC)
        src=tl.load(S+e,e<E,0);dst=tl.load(D+e,e<E,0)
        x=tl.load(X+src[:,None,None]*XDIM+XB+i[None,:,None]*C+c[None,None,:],
                  (e[:,None,None]<E)&(i[None,:,None]<I)&(c[None,None,:]<C),0.)
        v=tl.load(V+dst[:,None,None]*ODIM+OB+m[None,:,None]*C+c[None,None,:],
                  (e[:,None,None]<E)&(m[None,:,None]<OM)&(c[None,None,:]<C),0.)
        if MASK:v=v*tl.load(M+e,e<E,0.)[:,None,None]
        w=tl.load(W+PATH*C+c,c<C,0.)
        g=tl.load(G+(e[:,None]*P+PATH)*C+c[None,:],(e[:,None]<E)&(c[None,:]<C),0.)
        if (NEEDX or NEEDG) or NEEDW:
            b=tl.load(B+e[:,None,None]*BDIM+BB+i[None,:,None]*BS+m[None,None,:],
                      (e[:,None,None]<E)&(i[None,:,None]<I)&(m[None,None,:]<OM),0.)
            z=tl.sum(b[:,:,:,None]*v[:,None,:,:],axis=2)
            if NEEDX:
                dx=z*(w[None,None,:]*g[:,None,:])
                tl.atomic_add(DX+src[:,None,None]*XDIM+XB+i[None,:,None]*C+c[None,None,:],dx,
                    (e[:,None,None]<E)&(i[None,:,None]<I)&(c[None,None,:]<C),sem="relaxed")
            if NEEDG or NEEDW:
                value=tl.sum(x*z,axis=1);pointer=(e[:,None]*P+PATH)*C+c[None,:]
                active=(e[:,None]<E)&(c[None,:]<C)
                if NEEDG:tl.store(DG+pointer,value*w[None,:],active)
                if NEEDW:tl.store(DWE+pointer,value*g,active)
        if NEEDB:
            vw=(v*g[:,None,:])*w[None,None,:]
            db=tl.sum(x[:,:,None,:]*vw[:,None,:,:],axis=3)
            tl.store(DB+e[:,None,None]*BDIM+BB+i[None,:,None]*BS+m[None,None,:],db,
                     (e[:,None,None]<E)&(i[None,:,None]<I)&(m[None,None,:]<OM))


def forward_raw(xc,b,w,g,src,dst,mask,out,p):
    if p.get("path_math","ieee")!="ieee":raise ValueError("HP path messages require IEEE FP32 arithmetic")
    e=src.numel();be=p.get("path_tile_edges",1)
    for path,ni,xb,bb,bs,ob,om in p["path_specs"]:
        _forward_path[(triton.cdiv(e,be),)](xc,b,w,g,src,dst,mask,out,e,p["c"],p["xdim"],p["bdim"],
            p["odim"],p["p"],path,xb,bb,bs,ob,ni,om,bool(mask.numel()),be,triton.next_power_of_2(ni),
            triton.next_power_of_2(om),triton.next_power_of_2(p["c"]),num_warps=p.get("path_warps",4),enable_fp_fusion=True)


def adjoint_raw(xc,b,w,g,src,dst,mask,vc,dx,db,dg,dwe,p,needed):
    if p.get("path_math","ieee")!="ieee":raise ValueError("HP path messages require IEEE FP32 arithmetic")
    e=src.numel();be=p.get("path_tile_edges",1)
    for path,ni,xb,bb,bs,ob,om in p["path_specs"]:
        _adjoint_path[(triton.cdiv(e,be),)](xc,b,w,g,src,dst,mask,vc,dx if needed[0] else xc,
            db if needed[1] else b,dg if needed[3] else g,dwe if needed[2] else g,e,p["c"],p["xdim"],p["bdim"],
            p["odim"],p["p"],path,xb,bb,bs,ob,ni,om,needed[0],needed[1],needed[3],needed[2],bool(mask.numel()),
            be,triton.next_power_of_2(ni),triton.next_power_of_2(om),triton.next_power_of_2(p["c"]),
            num_warps=p.get("path_warps",4),enable_fp_fusion=True)
