"""Optional FP32 gather/angular/gate/scatter fusion for neutral MACE messages.

Only immutable layout metadata is planned. Topology, features, masks and weights
are live inputs. Mixed second derivatives use fused multilinear adjoints.
"""
import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch._subclasses.fake_tensor import is_fake
from ictc_o3.ops.angular_operator import angular_product
from ictc_o3.autograd.adjoint_requirements import required_gradients

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _feature_path(B,W,G,HB,HW,HG,M,V,e,i,c,dst,valid,C:tl.constexpr,BDIM:tl.constexpr,
                      ODIM:tl.constexpr,P:tl.constexpr,p:tl.constexpr,bb:tl.constexpr,
                      bs:tl.constexpr,ob:tl.constexpr,om:tl.constexpr,SECOND:tl.constexpr,
                      HASB:tl.constexpr,HASW:tl.constexpr,HASG:tl.constexpr,MASK:tl.constexpr,BLOCK:tl.constexpr,UNITW:tl.constexpr=False):
        w=1. if UNITW else tl.load(W+p*C+c,valid,0.);g=tl.load(G+(e*P+p)*C+c,valid,0.)
        magnetic=tl.full((BLOCK,),0,tl.float32);other=tl.full((BLOCK,),0,tl.float32)
        for m in tl.static_range(om):
            v=tl.load(V+dst*ODIM+ob+m*C+c,valid,0.)
            if MASK:v=v*tl.load(M+e,valid,0.)
            b=tl.load(B+e*BDIM+bb+i*bs+m,valid,0.)
            magnetic=magnetic+v*b
            if SECOND and HASB:other=other+v*tl.load(HB+e*BDIM+bb+i*bs+m,valid,0.)
        if SECOND:
            factor=tl.full((BLOCK,),0,tl.float32)
            if HASW:factor=factor+tl.load(HW+p*C+c,valid,0.)*g
            if HASG:factor=factor+w*tl.load(HG+(e*P+p)*C+c,valid,0.)
            value=factor*magnetic
            if HASB:value=value+(w*g)*other
        else:value=g*magnetic if UNITW else (w*g)*magnetic
        return value

    @triton.jit
    def _feature_static(B,W,G,HB,HW,HG,S,D,M,V,DX,E,C:tl.constexpr,XDIM:tl.constexpr,
                       BDIM:tl.constexpr,ODIM:tl.constexpr,P:tl.constexpr,XBASE:tl.constexpr,
                       I:tl.constexpr,PATHS:tl.constexpr,SECOND:tl.constexpr,HASB:tl.constexpr,
                       HASW:tl.constexpr,HASG:tl.constexpr,MASK:tl.constexpr,BLOCK:tl.constexpr,I32:tl.constexpr=False,UNITW:tl.constexpr=False):
        itype=tl.int32 if I32 else tl.int64
        index=tl.program_id(0).to(itype)*BLOCK+tl.arange(0,BLOCK)
        valid=index<E*I*C;e=index//(I*C);i=(index//C)%I;c=index%C
        src=tl.load(S+e,valid,0).to(itype);dst=tl.load(D+e,valid,0).to(itype)
        value=tl.full((BLOCK,),0,tl.float32)
        for k in tl.static_range(len(PATHS)):
            value=value+_feature_path(B,W,G,HB,HW,HG,M,V,e,i,c,dst,valid,C,BDIM,ODIM,P,
                PATHS.value[k][0],PATHS.value[k][1],PATHS.value[k][2],PATHS.value[k][3],PATHS.value[k][4],
                SECOND,HASB,HASW,HASG,MASK,BLOCK,UNITW)
        tl.atomic_add(DX+src*XDIM+XBASE+i*C+c,value,valid,sem="relaxed")

    @triton.jit
    def _feature_group(B,W,G,HB,HW,HG,S,D,M,V,DX,PM,E,C:tl.constexpr,XDIM:tl.constexpr,
                       BDIM:tl.constexpr,ODIM:tl.constexpr,P:tl.constexpr,XBASE:tl.constexpr,
                       I:tl.constexpr,K:tl.constexpr,SECOND:tl.constexpr,HASB:tl.constexpr,
                       HASW:tl.constexpr,HASG:tl.constexpr,MASK:tl.constexpr,BLOCK:tl.constexpr):
        index=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
        valid=index<E*I*C;e=index//(I*C);i=(index//C)%I;c=index%C
        src=tl.load(S+e,valid,0);dst=tl.load(D+e,valid,0)
        value=tl.full((BLOCK,),0,tl.float32)
        for k in tl.static_range(K):
            p=tl.load(PM+k);bb=tl.load(PM+K+k);bs=tl.load(PM+2*K+k);ob=tl.load(PM+3*K+k)
            v=tl.load(V+dst*ODIM+ob+c,valid,0.)
            if MASK:v=v*tl.load(M+e,valid,0.)
            w=tl.load(W+p*C+c,valid,0.);g=tl.load(G+(e*P+p)*C+c,valid,0.)
            b=tl.load(B+e*BDIM+bb+i*bs,valid,0.)
            if SECOND:
                factor=tl.full((BLOCK,),0,tl.float32)
                if HASW:factor=factor+tl.load(HW+p*C+c,valid,0.)*g
                if HASG:factor=factor+w*tl.load(HG+(e*P+p)*C+c,valid,0.)
                term=factor*b
                if HASB:term=term+(w*g)*tl.load(HB+e*BDIM+bb+i*bs,valid,0.)
                value=value+v*term
            else:value=value+((v*g)*w)*b
        tl.atomic_add(DX+src*XDIM+XBASE+i*C+c,value,valid,sem="relaxed")

    @triton.jit
    def _dx_irrep(B,W,G,S,D,M,V,DX,PM,E,C:tl.constexpr,XDIM:tl.constexpr,BDIM:tl.constexpr,
                  ODIM:tl.constexpr,P:tl.constexpr,XBASE:tl.constexpr,I:tl.constexpr,
                  K:tl.constexpr,
                  MASK:tl.constexpr,BLOCK:tl.constexpr):
        index=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
        valid=index<E*C*I;e=index//(C*I);c=(index//I)%C;i=index%I
        src=tl.load(S+e,valid,0);dst=tl.load(D+e,valid,0)
        value=tl.full((BLOCK,),0,tl.float32)
        for k in tl.static_range(K):
            p=tl.load(PM+k);bb=tl.load(PM+K+k);bs=tl.load(PM+2*K+k)
            ob=tl.load(PM+3*K+k);m=tl.load(PM+4*K+k)
            v=tl.load(V+dst*ODIM+ob+c*m,valid,0.)
            if MASK:v=v*tl.load(M+e,valid,0.)
            g=tl.load(G+e*P*C+p*C+c,valid,0.);w=tl.load(W+p*C+c,valid,0.)
            b=tl.load(B+e*BDIM+bb+i*bs,valid,0.)
            value=value+((v*g)*w)*b
        tl.atomic_add(DX+src*XDIM+XBASE+c*I+i,value,valid,sem="relaxed")

    @triton.jit
    def _forward(X,B,W,G,S,D,M,KM,O,E,C:tl.constexpr,XDIM:tl.constexpr,BDIM:tl.constexpr,
                 ODIM:tl.constexpr,P:tl.constexpr,K:tl.constexpr,IMAX:tl.constexpr,
                 MASK:tl.constexpr,BLOCK:tl.constexpr):
        index=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
        valid=index<E*C*K
        e,c,k=index//(C*K),(index//K)%C,index%K
        p=tl.load(KM+k);ni=tl.load(KM+K+k);xb=tl.load(KM+2*K+k)
        bb=tl.load(KM+3*K+k);bs=tl.load(KM+4*K+k)
        ob=tl.load(KM+5*K+k);m=tl.load(KM+6*K+k)
        src=tl.load(S+e,valid,0);dst=tl.load(D+e,valid,0)
        y=tl.full((BLOCK,),0,tl.float32)
        for i in range(IMAX):
            x=tl.load(X+src*XDIM+xb+c*ni+i,valid&(i<ni),0.)
            b=tl.load(B+e*BDIM+bb+i*bs,valid&(i<ni),0.)
            y=y+x*b
        w=tl.load(W+p*C+c,valid,0.);g=tl.load(G+e*P*C+p*C+c,valid,0.)
        value=(y*w)*g
        if MASK:value=value*tl.load(M+e,valid,0.)
        tl.atomic_add(O+dst*ODIM+ob+c*m,value,valid,sem="relaxed")

    @triton.jit
    def _db(X,W,G,S,D,M,BM,V,DB,E,C:tl.constexpr,XDIM:tl.constexpr,BDIM:tl.constexpr,
            ODIM:tl.constexpr,P:tl.constexpr,MASK:tl.constexpr,BC:tl.constexpr,BO:tl.constexpr):
        index=tl.program_id(0).to(tl.int64)*BO+tl.arange(0,BO)
        e,col=index//BDIM,index%BDIM;valid=index<E*BDIM
        p=tl.load(BM+col);xp=tl.load(BM+BDIM+col);ni=tl.load(BM+2*BDIM+col)
        ob=tl.load(BM+3*BDIM+col);m=tl.load(BM+4*BDIM+col)
        src=tl.load(S+e,valid,0);dst=tl.load(D+e,valid,0)
        c=tl.arange(0,BC);active=valid[None,:]&(c[:,None]<C)
        x=tl.load(X+src[None,:]*XDIM+xp[None,:]+c[:,None],active,0.)
        v=tl.load(V+dst[None,:]*ODIM+ob[None,:]+c[:,None],active,0.)
        if MASK:v=v*tl.load(M+e[None,:],valid[None,:],0.)
        g=tl.load(G+e[None,:]*P*C+p[None,:]*C+c[:,None],active,0.)
        w=tl.load(W+p[None,:]*C+c[:,None],active,0.)
        value=tl.sum(x*((v*g)*w),axis=0)
        tl.store(DB+index,value,valid)

    @triton.jit
    def _db_warp(X,W,G,S,D,M,BM,V,DB,E,C:tl.constexpr,XDIM:tl.constexpr,BDIM:tl.constexpr,
            ODIM:tl.constexpr,P:tl.constexpr,MASK:tl.constexpr,BO:tl.constexpr,I32:tl.constexpr=False,UNITW:tl.constexpr=False):
        itype=tl.int32 if I32 else tl.int64
        index=tl.program_id(0).to(itype)*BO+tl.arange(0,BO)
        e,col=index//BDIM,index%BDIM;valid=index<E*BDIM
        p=tl.load(BM+col);xp=tl.load(BM+BDIM+col)
        ob=tl.load(BM+3*BDIM+col)
        src=tl.load(S+e,valid,0).to(itype);dst=tl.load(D+e,valid,0).to(itype)
        lanes=tl.arange(0,32)
        total=tl.full((32,BO),0,tl.float32)
        for group in tl.static_range(triton.cdiv(C,32)):
            c=group*32+lanes;active=valid[None,:]&(c[:,None]<C)
            x=tl.load(X+src[None,:]*XDIM+xp[None,:]+c[:,None],active,0.)
            v=tl.load(V+dst[None,:]*ODIM+ob[None,:]+c[:,None],active,0.)
            if MASK:v=v*tl.load(M+e[None,:],valid[None,:],0.)
            g=tl.load(G+(e[None,:]*P+p[None,:])*C+c[:,None],active,0.)
            if UNITW:total=total+x*(v*g)
            else:
                w=tl.load(W+p[None,:]*C+c[:,None],active,0.)
                total=total+x*((v*g)*w)
        tl.store(DB+index,tl.sum(total,axis=0),valid)

    @triton.jit
    def _db_active_warp(X,W,G,S,D,M,BM,AM,V,DB,E,C:tl.constexpr,XDIM:tl.constexpr,BDIM:tl.constexpr,
            ODIM:tl.constexpr,P:tl.constexpr,A:tl.constexpr,MASK:tl.constexpr,BO:tl.constexpr):
        index=tl.program_id(0).to(tl.int64)*BO+tl.arange(0,BO)
        e=index//A;valid=index<E*A;col=tl.load(AM+index%A,valid,0)
        p=tl.load(BM+col);xp=tl.load(BM+BDIM+col);ob=tl.load(BM+3*BDIM+col)
        src=tl.load(S+e,valid,0);dst=tl.load(D+e,valid,0)
        lanes=tl.arange(0,32);total=tl.full((32,BO),0,tl.float32)
        for group in tl.static_range(triton.cdiv(C,32)):
            c=group*32+lanes;active=valid[None,:]&(c[:,None]<C)
            x=tl.load(X+src[None,:]*XDIM+xp[None,:]+c[:,None],active,0.)
            v=tl.load(V+dst[None,:]*ODIM+ob[None,:]+c[:,None],active,0.)
            if MASK:v=v*tl.load(M+e[None,:],valid[None,:],0.)
            g=tl.load(G+(e[None,:]*P+p[None,:])*C+c[:,None],active,0.)
            w=tl.load(W+p[None,:]*C+c[:,None],active,0.)
            total=total+x*((v*g)*w)
        tl.store(DB+e*BDIM+col,tl.sum(total,axis=0),valid)

    @triton.jit
    def _db_by_column(X,W,G,S,D,M,BM,V,DB,E,C:tl.constexpr,XDIM:tl.constexpr,BDIM:tl.constexpr,
            ODIM:tl.constexpr,P:tl.constexpr,MASK:tl.constexpr,BE:tl.constexpr):
        e=tl.program_id(0).to(tl.int64)*BE+tl.arange(0,BE);col=tl.program_id(1)
        valid=e<E;p=tl.load(BM+col);xp=tl.load(BM+BDIM+col);ob=tl.load(BM+3*BDIM+col)
        src=tl.load(S+e,valid,0);dst=tl.load(D+e,valid,0)
        lanes=tl.arange(0,32);total=tl.full((BE,32),0,tl.float32)
        for group in tl.static_range(triton.cdiv(C,32)):
            c=group*32+lanes;active=valid[:,None]&(c[None,:]<C)
            x=tl.load(X+src[:,None]*XDIM+xp+c[None,:],active,0.)
            v=tl.load(V+dst[:,None]*ODIM+ob+c[None,:],active,0.)
            if MASK:v=v*tl.load(M+e,valid,0.)[:,None]
            g=tl.load(G+(e[:,None]*P+p)*C+c[None,:],active,0.)
            w=tl.load(W+p*C+c,c<C,0.)
            total=total+x*((v*g)*w[None,:])
        tl.store(DB+col*E+e,tl.sum(total,axis=1),valid)

    @triton.jit
    def _dg(X,B,W,G,S,D,M,PM,V,DG,DWE,E,C:tl.constexpr,XDIM:tl.constexpr,BDIM:tl.constexpr,
            ODIM:tl.constexpr,P:tl.constexpr,IMAX:tl.constexpr,MMAX:tl.constexpr,
            MASK:tl.constexpr,NEEDG:tl.constexpr,NEEDW:tl.constexpr,BLOCK:tl.constexpr):
        index=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
        valid=index<E*P*C;e=index//(P*C);p=(index//C)%P;c=index%C
        ni=tl.load(PM+p);xb=tl.load(PM+P+p);bb=tl.load(PM+2*P+p)
        bs=tl.load(PM+3*P+p);ob=tl.load(PM+4*P+p);m=tl.load(PM+5*P+p)
        src=tl.load(S+e,valid,0);dst=tl.load(D+e,valid,0)
        value=tl.full((BLOCK,),0,tl.float32)
        for k in range(MMAX):
            active=valid&(k<m);y=tl.full((BLOCK,),0,tl.float32)
            for i in range(IMAX):
                x=tl.load(X+src*XDIM+xb+c*ni+i,active&(i<ni),0.)
                b=tl.load(B+e*BDIM+bb+i*bs+k,active&(i<ni),0.)
                y=y+x*b
            v=tl.load(V+dst*ODIM+ob+c*m+k,active,0.)
            if MASK:v=v*tl.load(M+e,active,0.)
            value=value+v*y
        if NEEDG:
            w=tl.load(W+p*C+c,valid,0.);tl.store(DG+index,value*w,valid)
        if NEEDW:
            g=tl.load(G+index,valid,0.);tl.store(DWE+index,value*g,valid)

    @triton.jit
    def _d2x_irrep(B,W,G,HB,HW,HG,S,D,M,V,DX,PM,E,C:tl.constexpr,XDIM:tl.constexpr,
                   BDIM:tl.constexpr,ODIM:tl.constexpr,P:tl.constexpr,XBASE:tl.constexpr,
                   I:tl.constexpr,K:tl.constexpr,HASB:tl.constexpr,HASW:tl.constexpr,
                   HASG:tl.constexpr,MASK:tl.constexpr,BLOCK:tl.constexpr):
        index=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
        valid=index<E*C*I;e=index//(C*I);c=(index//I)%C;i=index%I
        src=tl.load(S+e,valid,0);dst=tl.load(D+e,valid,0)
        value=tl.full((BLOCK,),0,tl.float32)
        for k in tl.static_range(K):
            p=tl.load(PM+k);bb=tl.load(PM+K+k);bs=tl.load(PM+2*K+k)
            ob=tl.load(PM+3*K+k);m=tl.load(PM+4*K+k)
            v=tl.load(V+dst*ODIM+ob+c*m,valid,0.)
            if MASK:v=v*tl.load(M+e,valid,0.)
            w=tl.load(W+p*C+c,valid,0.);g=tl.load(G+e*P*C+p*C+c,valid,0.)
            factor=tl.full((BLOCK,),0,tl.float32)
            if HASW:factor=factor+tl.load(HW+p*C+c,valid,0.)*g
            if HASG:factor=factor+w*tl.load(HG+e*P*C+p*C+c,valid,0.)
            b=tl.load(B+e*BDIM+bb+i*bs,valid,0.)
            term=factor*b
            if HASB:term=term+(w*g)*tl.load(HB+e*BDIM+bb+i*bs,valid,0.)
            value=value+v*term
        tl.atomic_add(DX+src*XDIM+XBASE+c*I+i,value,valid,sem="relaxed")

    @triton.jit
    def _d2b(X,W,G,HX,HW,HG,S,D,M,BM,V,DB,E,C:tl.constexpr,XDIM:tl.constexpr,
             BDIM:tl.constexpr,ODIM:tl.constexpr,P:tl.constexpr,HASX:tl.constexpr,
             HASW:tl.constexpr,HASG:tl.constexpr,MASK:tl.constexpr,BC:tl.constexpr,BO:tl.constexpr):
        index=tl.program_id(0).to(tl.int64)*BO+tl.arange(0,BO)
        e,col=index//BDIM,index%BDIM;valid=index<E*BDIM
        p=tl.load(BM+col);xp=tl.load(BM+BDIM+col);ni=tl.load(BM+2*BDIM+col)
        ob=tl.load(BM+3*BDIM+col);m=tl.load(BM+4*BDIM+col)
        src=tl.load(S+e,valid,0);dst=tl.load(D+e,valid,0)
        c=tl.arange(0,BC);active=valid[None,:]&(c[:,None]<C)
        x=tl.load(X+src[None,:]*XDIM+xp[None,:]+c[:,None],active,0.)
        v=tl.load(V+dst[None,:]*ODIM+ob[None,:]+c[:,None],active,0.)
        if MASK:v=v*tl.load(M+e[None,:],valid[None,:],0.)
        g=tl.load(G+e[None,:]*P*C+p[None,:]*C+c[:,None],active,0.)
        w=tl.load(W+p[None,:]*C+c[:,None],active,0.)
        factor=tl.full((BC,BO),0,tl.float32)
        if HASW:factor=factor+tl.load(HW+p[None,:]*C+c[:,None],active,0.)*g
        if HASG:factor=factor+w*tl.load(HG+e[None,:]*P*C+p[None,:]*C+c[:,None],active,0.)
        term=factor*x
        if HASX:term=term+(w*g)*tl.load(HX+src[None,:]*XDIM+xp[None,:]+c[:,None],active,0.)
        tl.store(DB+index,tl.sum(v*term,axis=0),valid)

    @triton.jit
    def _d2gw(X,B,W,G,HX,HB,HW,HG,S,D,M,PM,V,DG,DWE,E,C:tl.constexpr,
              XDIM:tl.constexpr,BDIM:tl.constexpr,ODIM:tl.constexpr,P:tl.constexpr,
              IMAX:tl.constexpr,MMAX:tl.constexpr,HASX:tl.constexpr,HASB:tl.constexpr,
              HASW:tl.constexpr,HASG:tl.constexpr,MASK:tl.constexpr,NEEDG:tl.constexpr,
              NEEDW:tl.constexpr,BLOCK:tl.constexpr):
        index=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
        valid=index<E*P*C;e=index//(P*C);p=(index//C)%P;c=index%C
        ni=tl.load(PM+p);xb=tl.load(PM+P+p);bb=tl.load(PM+2*P+p)
        bs=tl.load(PM+3*P+p);ob=tl.load(PM+4*P+p);m=tl.load(PM+5*P+p)
        src=tl.load(S+e,valid,0);dst=tl.load(D+e,valid,0)
        value=tl.full((BLOCK,),0,tl.float32);cross=tl.full((BLOCK,),0,tl.float32)
        for k in range(MMAX):
            active=valid&(k<m);y=tl.full((BLOCK,),0,tl.float32);hy=tl.full((BLOCK,),0,tl.float32)
            for i in range(IMAX):
                use=active&(i<ni)
                x=tl.load(X+src*XDIM+xb+c*ni+i,use,0.)
                b=tl.load(B+e*BDIM+bb+i*bs+k,use,0.)
                y=y+x*b
                if HASX:hy=hy+tl.load(HX+src*XDIM+xb+c*ni+i,use,0.)*b
                if HASB:hy=hy+x*tl.load(HB+e*BDIM+bb+i*bs+k,use,0.)
            v=tl.load(V+dst*ODIM+ob+c*m+k,active,0.)
            if MASK:v=v*tl.load(M+e,active,0.)
            value=value+v*y;cross=cross+v*hy
        if NEEDG:
            w=tl.load(W+p*C+c,valid,0.);dg=w*cross
            if HASW:dg=dg+tl.load(HW+p*C+c,valid,0.)*value
            tl.store(DG+index,dg,valid)
        if NEEDW:
            g=tl.load(G+index,valid,0.);dw=g*cross
            if HASG:dw=dw+tl.load(HG+index,valid,0.)*value
            tl.store(DWE+index,dw,valid)

    @triton.jit
    def _d2v(X,B,W,G,HX,HB,HW,HG,S,D,M,KM,O,E,C:tl.constexpr,XDIM:tl.constexpr,
             BDIM:tl.constexpr,ODIM:tl.constexpr,P:tl.constexpr,K:tl.constexpr,
             IMAX:tl.constexpr,HASX:tl.constexpr,HASB:tl.constexpr,HASW:tl.constexpr,
             HASG:tl.constexpr,MASK:tl.constexpr,BLOCK:tl.constexpr):
        index=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
        valid=index<E*C*K;e=index//(C*K);c=(index//K)%C;k=index%K
        p=tl.load(KM+k);ni=tl.load(KM+K+k);xb=tl.load(KM+2*K+k)
        bb=tl.load(KM+3*K+k);bs=tl.load(KM+4*K+k)
        ob=tl.load(KM+5*K+k);m=tl.load(KM+6*K+k)
        src=tl.load(S+e,valid,0);dst=tl.load(D+e,valid,0)
        y=tl.full((BLOCK,),0,tl.float32);hy=tl.full((BLOCK,),0,tl.float32)
        for i in range(IMAX):
            use=valid&(i<ni)
            x=tl.load(X+src*XDIM+xb+c*ni+i,use,0.)
            b=tl.load(B+e*BDIM+bb+i*bs,use,0.)
            y=y+x*b
            if HASX:hy=hy+tl.load(HX+src*XDIM+xb+c*ni+i,use,0.)*b
            if HASB:hy=hy+x*tl.load(HB+e*BDIM+bb+i*bs,use,0.)
        w=tl.load(W+p*C+c,valid,0.);g=tl.load(G+e*P*C+p*C+c,valid,0.)
        factor=tl.full((BLOCK,),0,tl.float32)
        if HASW:factor=factor+tl.load(HW+p*C+c,valid,0.)*g
        if HASG:factor=factor+w*tl.load(HG+e*P*C+p*C+c,valid,0.)
        value=factor*y+(w*g)*hy
        if MASK:value=value*tl.load(M+e,valid,0.)
        tl.atomic_add(O+dst*ODIM+ob+c*m,value,valid,sem="relaxed")

    @triton.jit
    def _gate_group(X,B,W,G,HX,HB,HW,HG,S,D,M,PM,V,DG,DWE,E,C:tl.constexpr,
                    XDIM:tl.constexpr,BDIM:tl.constexpr,ODIM:tl.constexpr,P:tl.constexpr,
                    NP:tl.constexpr,I:tl.constexpr,OM:tl.constexpr,SECOND:tl.constexpr,
                    HASX:tl.constexpr,HASB:tl.constexpr,HASW:tl.constexpr,HASG:tl.constexpr,
                    MASK:tl.constexpr,NEEDG:tl.constexpr,NEEDW:tl.constexpr,BLOCK:tl.constexpr,I32:tl.constexpr=False,UNITW:tl.constexpr=False):
        itype=tl.int32 if I32 else tl.int64
        index=tl.program_id(0).to(itype)*BLOCK+tl.arange(0,BLOCK)
        valid=index<E*NP*C;e=index//(NP*C);pindex=(index//C)%NP;c=index%C
        p=tl.load(PM+pindex);xb=tl.load(PM+NP+pindex);bb=tl.load(PM+2*NP+pindex)
        bs=tl.load(PM+3*NP+pindex);ob=tl.load(PM+4*NP+pindex)
        src=tl.load(S+e,valid,0).to(itype);dst=tl.load(D+e,valid,0).to(itype)
        value=tl.full((BLOCK,),0,tl.float32);cross=tl.full((BLOCK,),0,tl.float32)
        for i in tl.static_range(I):
            x=tl.load(X+src*XDIM+xb+i*C+c,valid,0.)
            if SECOND and HASX:hx=tl.load(HX+src*XDIM+xb+i*C+c,valid,0.)
            for m in tl.static_range(OM):
                b=tl.load(B+e*BDIM+bb+i*bs+m,valid,0.)
                v=tl.load(V+dst*ODIM+ob+m*C+c,valid,0.)
                if MASK:v=v*tl.load(M+e,valid,0.)
                value=value+(v*x)*b
                if SECOND:
                    term=tl.full((BLOCK,),0,tl.float32)
                    if HASX:term=term+hx*b
                    if HASB:term=term+x*tl.load(HB+e*BDIM+bb+i*bs+m,valid,0.)
                    cross=cross+v*term
        output=(e*P+p)*C+c
        if NEEDG:
            w=1. if UNITW else tl.load(W+p*C+c,valid,0.)
            dg=w*cross if SECOND else w*value
            if SECOND and HASW:dg=dg+tl.load(HW+p*C+c,valid,0.)*value
            tl.store(DG+output,dg,valid)
        if NEEDW:
            g=tl.load(G+output,valid,0.)
            dw=g*cross if SECOND else g*value
            if SECOND and HASG:dw=dw+tl.load(HG+output,valid,0.)*value
            tl.store(DWE+output,dw,valid)

    @triton.jit
    def _message_group(X,B,W,G,HX,HB,HW,HG,S,D,M,PM,O,E,C:tl.constexpr,
                       XDIM:tl.constexpr,BDIM:tl.constexpr,ODIM:tl.constexpr,P:tl.constexpr,
                       NP:tl.constexpr,I:tl.constexpr,OM:tl.constexpr,SECOND:tl.constexpr,
                       HASX:tl.constexpr,HASB:tl.constexpr,HASW:tl.constexpr,HASG:tl.constexpr,
                       MASK:tl.constexpr,BLOCK:tl.constexpr,I32:tl.constexpr=False,UNITW:tl.constexpr=False):
        itype=tl.int32 if I32 else tl.int64
        index=tl.program_id(0).to(itype)*BLOCK+tl.arange(0,BLOCK)
        valid=index<E*NP*OM*C;e=index//(NP*OM*C);pi=(index//(OM*C))%NP;m=(index//C)%OM;c=index%C
        p=tl.load(PM+pi);xb=tl.load(PM+NP+pi);bb=tl.load(PM+2*NP+pi)
        bs=tl.load(PM+3*NP+pi);ob=tl.load(PM+4*NP+pi)
        src=tl.load(S+e,valid,0).to(itype);dst=tl.load(D+e,valid,0).to(itype)
        y=tl.full((BLOCK,),0,tl.float32);hy=tl.full((BLOCK,),0,tl.float32)
        for i in tl.static_range(I):
            x=tl.load(X+src*XDIM+xb+i*C+c,valid,0.)
            b=tl.load(B+e*BDIM+bb+i*bs+m,valid,0.)
            y=y+x*b
            if SECOND and HASX:hy=hy+tl.load(HX+src*XDIM+xb+i*C+c,valid,0.)*b
            if SECOND and HASB:hy=hy+x*tl.load(HB+e*BDIM+bb+i*bs+m,valid,0.)
        w=1. if UNITW else tl.load(W+p*C+c,valid,0.);g=tl.load(G+(e*P+p)*C+c,valid,0.)
        if SECOND:
            factor=tl.full((BLOCK,),0,tl.float32)
            if HASW:factor=factor+tl.load(HW+p*C+c,valid,0.)*g
            if HASG:factor=factor+w*tl.load(HG+(e*P+p)*C+c,valid,0.)
            value=factor*y+(w*g)*hy
        else:value=y*g if UNITW else (y*w)*g
        if MASK:value=value*tl.load(M+e,valid,0.)
        tl.atomic_add(O+dst*ODIM+ob+m*C+c,value,valid,sem="relaxed")

    @triton.jit
    def _bond_group(X,W,G,HX,HW,HG,S,D,M,KM,V,O,E,C:tl.constexpr,XDIM:tl.constexpr,
                    ODIM:tl.constexpr,BDIM:tl.constexpr,P:tl.constexpr,XB:tl.constexpr,BB:tl.constexpr,
                    I:tl.constexpr,K:tl.constexpr,SECOND:tl.constexpr,HASX:tl.constexpr,
                    HASW:tl.constexpr,HASG:tl.constexpr,MASK:tl.constexpr,BI:tl.constexpr,BC:tl.constexpr,BK:tl.constexpr):
        e=tl.program_id(0).to(tl.int64);k=tl.program_id(1)*BK+tl.arange(0,BK)
        c=tl.arange(0,BC);i=tl.arange(0,BI)
        src=tl.load(S+e);dst=tl.load(D+e)
        path=tl.load(KM+k,k<K,0);ob=tl.load(KM+K+k,k<K,0)
        active=(c[:,None]<C)&(k[None,:]<K)
        v=tl.load(V+dst*ODIM+ob[None,:]+c[:,None],active,0.)
        if MASK:v=v*tl.load(M+e)
        g=tl.load(G+(e*P+path[None,:])*C+c[:,None],active,0.)
        w=tl.load(W+path[None,:]*C+c[:,None],active,0.)
        x=tl.load(X+src*XDIM+XB+i[:,None]*C+c[None,:],(i[:,None]<I)&(c[None,:]<C),0.)
        if SECOND:
            factor=tl.full((BC,BK),0,tl.float32)
            if HASW:factor=factor+tl.load(HW+path[None,:]*C+c[:,None],active,0.)*g
            if HASG:factor=factor+w*tl.load(HG+(e*P+path[None,:])*C+c[:,None],active,0.)
            product=x[:,:,None]*(v*factor)[None,:,:]
            if HASX:
                hx=tl.load(HX+src*XDIM+XB+i[:,None]*C+c[None,:],(i[:,None]<I)&(c[None,:]<C),0.)
                product=product+hx[:,:,None]*((v*g)*w)[None,:,:]
        else:product=x[:,:,None]*((v*g)*w)[None,:,:]
        result=tl.sum(product,1)
        tl.store(O+e*BDIM+BB+i[:,None]*K+k[None,:],result,(i[:,None]<I)&(k[None,:]<K))


def _plan(tp,keys,device,grouped_forward=False):
    cache=getattr(tp,"_message_layout_cache",None)
    if cache is None:cache={};tp._message_layout_cache=cache
    cache_key=(tuple(keys),str(device),bool(grouped_forward))
    if cache_key in cache and "dx_static_groups" in cache[cache_key] and "path_specs" in cache[cache_key]:return cache[cache_key]
    c=tp.channels;xoffset={};xdim=0
    for l in keys:
        ni=2*l+1;xoffset[l]=xdim
        xdim+=c*ni
    ooffset={};odim=0
    for l in range(tp.lmax+1):
        ooffset[l]=odim;odim+=c*tp.path_counts_by_l.get(l,0)*(2*l+1)
    km=[];bm=[];bm_channel=[];pm=[None]*tp.num_paths;groups=[];bdim=0
    for group in tp._groups:
        l1=int(group["l1"]);ni=2*l1+1;kt=int(group["k_total"]);xb=xoffset[l1]
        begin=len(km)
        for p,l,s,end in group["segments"]:
            p,l,s,end=map(int,(p,l,s,end));m=2*l+1
            ob=ooffset[l]+int(tp.path_offset[p])*c*m
            pm[p]=(ni,xb,bdim+s,kt,ob,m)
            for k in range(s,end):km.append((p,ni,xb,bdim+k,kt,ob+k-s,m))
        for i in range(ni):
            for k in range(kt):
                p,_,_,_,_,ob,m=km[begin+k]
                bm.append((p,xb+i,ni,ob,m))
                _,l,start,end=next(segment for segment in group["segments"] if int(segment[0])==p)
                base=ooffset[int(l)]+int(tp.path_offset[p])*c*m
                bm_channel.append((p,xb+i*c,ni,base+(k-int(start))*c,m))
        groups.append((xb,ni,bdim,kt,list(group["segments"])))
        bdim+=ni*kt
    def tensor(rows,columns=None):
        if not rows:return torch.empty(columns or 0,0,device=device,dtype=torch.int32)
        return torch.tensor(rows,device=device,dtype=torch.int32).T.contiguous()
    result={"km":tensor(km),"bm":tensor(bm),"bm_channel":tensor(bm_channel),"pm":tensor(pm),"groups":groups,
            "bm_static":tuple(bm_channel),
            "path_specs":tuple((path,*row) for path,row in enumerate(pm)),
            "grouped_forward":bool(grouped_forward),
            "xdim":xdim,"bdim":bdim,"odim":odim,"k":len(km),"c":c,"p":tp.num_paths,
            "imax":max(row[0] for row in pm),"mmax":max(row[-1] for row in pm),
            "ooffset":ooffset,"path_offset":list(tp.path_offset),"lmax":tp.lmax,
            "counts":dict(tp.path_counts_by_l)}
    result["dx_groups"]=[(xoffset[l],2*l+1,tensor([(row[0],row[3],row[4],row[5],row[6]) for row in km if row[2]==xoffset[l]],5)) for l in keys]
    output_meta={int(p):(ooffset[int(l)]+int(tp.path_offset[int(p)])*c*(2*int(l)+1),2*int(l)+1)
                 for group in tp._groups for p,l,_,_ in group["segments"]}
    result["km_channel"]=tensor([(row[0],row[1],row[2],row[3],row[4],
        output_meta[row[0]][0]+(row[5]-output_meta[row[0]][0])*c,row[6]) for row in km],7)
    component_paths=[0]*(odim//c)
    for path,(base,m) in output_meta.items():
        for mag in range(m):component_paths[base//c+mag]=path
    result["component_paths"]=torch.tensor(component_paths,device=device,dtype=torch.int32)
    result["dx_channel_groups"]=[(xoffset[l],2*l+1,tensor([(row[0],row[3],row[4],
        output_meta[row[0]][0]+(row[5]-output_meta[row[0]][0])*c,row[6]) for row in km if row[2]==xoffset[l]],5)) for l in keys]
    gate_groups={}
    for p,(ni,xb,bb,bs,ob,m) in enumerate(pm):
        gate_groups.setdefault((ni,m),[]).append((p,xb,bb,bs,ob))
    result["gate_groups"]=[(ni,m,tensor(rows,5)) for (ni,m),rows in gate_groups.items()]
    result["dx_static_groups"]=[(xoffset[l],2*l+1,tuple(
        (p,row[2],row[3],row[4],row[5]) for p,row in enumerate(pm) if row[1]==xoffset[l])) for l in keys]
    result["bond_groups"]=[]
    for xb,ni,bb,kt,segments in groups:
        rows=[]
        for p,l,start,end in segments:
            base=ooffset[int(l)]+int(tp.path_offset[int(p)])*c*(2*int(l)+1)
            rows.extend((int(p),base+(k-int(start))*c) for k in range(int(start),int(end)))
        result["bond_groups"].append((xb,ni,bb,kt,tensor(rows,2)))
    cache[cache_key]=result
    return result


def _native(x,b,w,g,src,dst,mask,plan):
    n,c=x.shape[0],plan["c"];e=src.numel();parts={l:[None]*plan["counts"].get(l,0) for l in range(plan["lmax"]+1)}
    # This differentiable fallback is used by higher coordinate adjoints.
    # DTK's installed Triton backend cannot compile gfx926; native torch matmul
    # preserves the same bilinear expression and its higher derivatives.
    angular_backend = "triton_full"
    if torch.version.hip and x.is_cuda and getattr(
            torch.cuda.get_device_properties(x.device), 'gcnArchName', '').split(':')[0] == 'gfx926':
        angular_backend = "torch"
    for xb,ni,bb,kt,segments in plan["groups"]:
        node=x[:,xb:xb+c*ni].reshape(n,c,ni)
        magnetic=b[:,bb:bb+ni*kt].reshape(e,ni,kt)
        y=angular_product(node[src],magnetic,angular_backend)
        for p,l,start,end in segments:
            edge=(y[...,start:end]*w[p].view(1,c,1))*g[:,p,:].unsqueeze(-1)
            if mask.numel():edge=edge*mask.view(e,1,1)
            aggregate=edge.new_zeros(n,c,end-start).index_add(0,dst,edge)
            parts[l][plan["path_offset"][p]]=aggregate
    return torch.cat([torch.cat(parts[l],dim=1).reshape(n,-1) for l in parts if parts[l]],dim=1)


def _forward_raw(x,b,w,g,src,dst,mask,p):
    if p.get("grouped_forward",False):return _group_message_raw(x,b,w,g,src,dst,mask,p)
    out=x.new_zeros(x.shape[0],p["odim"])
    if src.numel():
        _forward[(triton.cdiv(src.numel()*p["c"]*p["k"],256),)](
            x,b,w,g,src,dst,mask,p["km"],out,src.numel(),p["c"],p["xdim"],p["bdim"],
            p["odim"],p["p"],p["k"],p["imax"],bool(mask.numel()),256,num_warps=4,enable_fp_fusion=True)
    return out


def _normal_message_output(out,p):
    if p.get("generated_layout","packed")=="fused_pack":
        from ictc_o3.ops.message_layout import output_copy
        from ictc_o3.ops.message_layout import convert
        if convert is not None:return output_copy(out,p,to_packed=False)
    n=out.shape[0];c=p["c"]
    return torch.cat([out[:,p["ooffset"][l]:p["ooffset"][l]+c*p["counts"].get(l,0)*(2*l+1)].reshape(
        n,p["counts"][l],2*l+1,c).transpose(2,3).reshape(n,c*p["counts"][l]*(2*l+1))
        for l in range(p["lmax"]+1) if p["counts"].get(l,0)],dim=1).contiguous()


def _group_message_raw(x,b,w,g,src,dst,mask,p,incoming=None,*,output=None,channel_node=None,
                       channel_tangent=None):
    if "destination_csr" in p:
        from ictc_o3.ops.csr_group_message import destination_raw
        return destination_raw(x,b,w,g,src,mask,p,incoming)
    n=x.shape[0];c=p["c"];e=src.numel();out=x.new_zeros(n,p["odim"]) if output is None else output
    if e:
        i32=_int32_safe(x,e,p)
        xc=_channel_node(x,p) if channel_node is None else channel_node
        if p.get("fused_path_forward",False) and incoming is None:
            from ictc_o3.ops.fused_path_message import forward_raw
            forward_raw(xc,b,w,g,src,dst,mask,out,p)
            return _normal_message_output(out,p) if output is None else out
        if incoming is None:flags=(False,)*4;hxc,hb,hw,hg=xc,b,w,g
        else:
            flags=tuple(value is not None for value in incoming)
            hx,hb,hw,hg=[value.contiguous() if value is not None else original for value,original in zip(incoming,(x,b,w,g))]
            hxc=(_channel_node(hx,p) if channel_tangent is None else channel_tangent) if flags[0] else xc
        for ni,m,paths in p["gate_groups"]:
            block=p.get("forward_block",512 if c>=64 and m>1 and incoming is None else 256)
            _message_group[(triton.cdiv(e*paths.shape[1]*m*c,block),)](xc,b,w,g,hxc,hb,hw,hg,src,dst,mask,paths,out,
                e,c,p["xdim"],p["bdim"],p["odim"],p["p"],paths.shape[1],ni,m,incoming is not None,
                *flags,bool(mask.numel()),block,I32=i32,UNITW=p.get("unit_weights",False),num_warps=p.get("forward_warps",4),enable_fp_fusion=True)
    return _normal_message_output(out,p) if output is None else out


def _vjp_raw(x,b,w,g,src,dst,mask,grad,p,needed,*,feature_output=None,gate_output=None,channel_inputs=None,
             harmonic_output=None,projection=None):
    grad=grad.contiguous();e=src.numel();has_mask=bool(mask.numel())
    dx=(torch.zeros_like(x) if feature_output is None else feature_output) if needed[0] else None
    direct=harmonic_output is not None and needed[1] and p.get("direct_harmonic",False)
    db=(b.new_empty(p["bdim"],e).T if p.get("geometry_by_column",False) else torch.empty_like(b)) if needed[1] and not direct else None
    dg=(torch.empty_like(g) if gate_output is None else gate_output) if needed[3] else None
    dwe=torch.empty_like(g) if needed[2] else None
    if e:
        i32=_int32_safe(x,e,p)
        xc=vc=None
        if channel_inputs is not None:xc,vc=channel_inputs
        elif needed[1] or needed[2] or needed[3]:xc,vc=_channel_inputs(x,grad,p)
        elif needed[0]:vc=_channel_cotangent(grad,p)
        if p.get("fused_path_adjoint",False) and any(needed):
            from ictc_o3.ops.fused_path_message import adjoint_raw
            if xc is None:xc=_channel_node(x,p)
            adjoint_raw(xc,b,w,g,src,dst,mask,vc,dx,db,dg,dwe,p,needed)
            dw=dwe.sum(0) if needed[2] else None
            if needed[0] and feature_output is None:dx=_normal_feature_gradient(dx,p)
            return dx,db,dw,dg
        if needed[0]:
            if "destination_csr" in p:
                from ictc_o3.ops.csr_group_message import source_raw
                source_raw(b,w,g,src,dst,mask,vc,dx,p)
            elif p["c"]>=64:
                block=p.get("feature_block",256)
                for xb,ni,paths in p["dx_static_groups"]:
                    _feature_static[(triton.cdiv(e*p["c"]*ni,block),)](b,w,g,b,w,g,src,dst,mask,vc,dx,
                        e,p["c"],p["xdim"],p["bdim"],p["odim"],p["p"],xb,ni,paths,
                        False,False,False,False,has_mask,block,I32=i32,UNITW=p.get("unit_weights",False),num_warps=p.get("feature_warps",4),enable_fp_fusion=True)
            else:
                for xb,ni,paths in p["dx_channel_groups"]:
                    _feature_group[(triton.cdiv(e*p["c"]*ni,256),)](b,w,g,b,w,g,src,dst,mask,vc,dx,paths,
                        e,p["c"],p["xdim"],p["bdim"],p["odim"],p["p"],xb,ni,
                        paths.shape[1],False,False,False,False,has_mask,256,num_warps=4,enable_fp_fusion=True)
        if needed[1]:
            if p.get("cached_edge_geometry",False):
                from ictc_o3.backends.k100.cached_edge_geometry import cached_geometry
                db=cached_geometry(xc,w,g,src,dst,mask,vc,p)
            elif direct:
                from ictc_o3.ops.direct_harmonic_gradient import direct_harmonic_raw
                direct_harmonic_raw(xc,w,g,src,dst,mask,vc,harmonic_output,p,projection)
            elif p.get("geometry_by_column",False):
                be=p.get("geometry_tile_edges",32)
                _db_by_column[(triton.cdiv(e,be),p["bdim"])](xc,w,g,src,dst,mask,p["bm_channel"],vc,db,e,p["c"],p["xdim"],
                    p["bdim"],p["odim"],p["p"],has_mask,be,num_warps=4,enable_fp_fusion=True)
            elif p["c"]>=64:
                _db_warp[(triton.cdiv(e*p["bdim"],32),)](xc,w,g,src,dst,mask,p["bm_channel"],vc,db,e,p["c"],p["xdim"],
                    p["bdim"],p["odim"],p["p"],has_mask,32,I32=i32,UNITW=p.get("unit_weights",False),num_warps=4,enable_fp_fusion=True)
            else:
                _db[(triton.cdiv(e*p["bdim"],16),)](xc,w,g,src,dst,mask,p["bm_channel"],vc,db,e,p["c"],p["xdim"],
                    p["bdim"],p["odim"],p["p"],has_mask,triton.next_power_of_2(p["c"]),16,num_warps=4,enable_fp_fusion=True)
        if needed[2] or needed[3]:
            block=p.get("gate_block",256)
            for ni,m,paths in p["gate_groups"]:
                _gate_group[(triton.cdiv(e*paths.shape[1]*p["c"],block),)](xc,b,w,g,xc,b,w,g,src,dst,mask,paths,vc,
                    dg if needed[3] else g,dwe if needed[2] else g,e,p["c"],p["xdim"],p["bdim"],p["odim"],
                    p["p"],paths.shape[1],ni,m,False,False,False,False,False,has_mask,needed[3],needed[2],
                    block,I32=i32,UNITW=p.get("unit_weights",False),num_warps=p.get("gate_warps",4),enable_fp_fusion=True)
    dw=dwe.sum(0) if needed[2] else None
    if needed[0] and feature_output is None:dx=_normal_feature_gradient(dx,p)
    return dx,db,dw,dg


def _int32_safe(x,edges,p):
    return bool(p.get("int32_index",False) and max(x.numel(),x.shape[0]*p["odim"],
        edges*p["bdim"],edges*p["p"]*p["c"],edges*p["k"]*p["c"])<(1<<31))


def _channel_node(x,p):
    if p.get("generated_layout","packed")=="fused_pack":
        from ictc_o3.ops.message_layout import feature_copy
        from ictc_o3.ops.message_layout import convert
        if convert is not None:return feature_copy(x,p,to_packed=True)
    n=x.shape[0];c=p["c"]
    return torch.cat([x[:,xb:xb+c*ni].reshape(n,c,ni).transpose(1,2).reshape(n,c*ni)
                  for xb,ni,_ in p["dx_groups"]],dim=1).contiguous()


def _normal_feature_gradient(dx,p):
    if p.get("generated_layout","packed")=="fused_pack":
        from ictc_o3.ops.message_layout import feature_copy
        from ictc_o3.ops.message_layout import convert
        if convert is not None:return feature_copy(dx,p,to_packed=False)
    n=dx.shape[0];c=p["c"]
    return torch.cat([dx[:,xb:xb+c*ni].reshape(n,ni,c).transpose(1,2).reshape(n,c*ni)
                      for xb,ni,_ in p["dx_groups"]],dim=1).contiguous()


def _channel_inputs(x,grad,p):
    xc=_channel_node(x,p)
    return xc,_channel_cotangent(grad,p)


def _channel_cotangent(grad,p):
    if p.get("generated_layout","packed")=="fused_pack":
        from ictc_o3.ops.message_layout import output_copy
        from ictc_o3.ops.message_layout import convert
        if convert is not None:return output_copy(grad,p,to_packed=True)
    n=grad.shape[0];c=p["c"]
    vc=torch.cat([grad[:,p["ooffset"][l]:p["ooffset"][l]+c*p["counts"].get(l,0)*(2*l+1)].reshape(
        n,p["counts"][l],c,2*l+1).transpose(2,3).reshape(n,-1)
        for l in range(p["lmax"]+1) if p["counts"].get(l,0)],dim=1).contiguous()
    return vc


def _bond_raw(xc,w,g,hxc,hw,hg,src,dst,mask,vc,output,p,second,flags):
    e=src.numel();c=p["c"]
    for xb,ni,bb,kt,paths in p["bond_groups"]:
        bi=triton.next_power_of_2(ni);bc=triton.next_power_of_2(c)
        _bond_group[(e,triton.cdiv(kt,8))](xc,w,g,hxc,hw,hg,src,dst,mask,paths,vc,output,
            e,c,p["xdim"],p["odim"],p["bdim"],p["p"],xb,bb,ni,kt,second,
            flags[0],flags[2],flags[3],bool(mask.numel()),bi,bc,8,
            num_warps=4 if bi*bc*8<=2048 else 8,enable_fp_fusion=True)


def _second_raw(x,b,w,g,src,dst,mask,grad,p,incoming,needed,need_grad,*,feature_output=None,
                gate_output=None,cotangent_output=None,channel_inputs=None):
    hx,hb,hw,hg=[value.contiguous() if value is not None else original
                 for value,original in zip(incoming,(x,b,w,g))]
    flags=tuple(value is not None for value in incoming)
    if not any(flags):return None,None,None,None,None
    grad=grad.contiguous();e=src.numel();has_mask=bool(mask.numel())
    dx=(torch.zeros_like(x) if feature_output is None else feature_output) if needed[0] else None
    db=torch.empty_like(b) if needed[1] else None
    dg=(torch.empty_like(g) if gate_output is None else gate_output) if needed[3] else None
    dwe=torch.empty_like(g) if needed[2] else None
    dy=(x.new_zeros(x.shape[0],p["odim"]) if cotangent_output is None else cotangent_output) if need_grad else None
    if e:
        xc=vc=hxc=None
        if channel_inputs is not None:xc,vc,hxc=channel_inputs
        elif needed[1] or needed[2] or needed[3]:
            xc,vc=_channel_inputs(x,grad,p)
            hxc=_channel_node(hx,p) if flags[0] else xc
        elif needed[0]:vc=_channel_cotangent(grad,p)
        if needed[0]:
            if "destination_csr" in p:
                from ictc_o3.ops.csr_group_message import source_raw
                source_raw(b,w,g,src,dst,mask,vc,dx,p,incoming)
            elif p["c"]>=64:
                for xb,ni,paths in p["dx_static_groups"]:
                    _feature_static[(triton.cdiv(e*p["c"]*ni,256),)](b,w,g,hb,hw,hg,src,dst,mask,vc,dx,
                        e,p["c"],p["xdim"],p["bdim"],p["odim"],p["p"],xb,ni,paths,
                        True,*flags[1:],has_mask,256,num_warps=4,enable_fp_fusion=True)
            else:
                for xb,ni,paths in p["dx_channel_groups"]:
                    _feature_group[(triton.cdiv(e*p["c"]*ni,256),)](b,w,g,hb,hw,hg,src,dst,mask,vc,dx,paths,
                        e,p["c"],p["xdim"],p["bdim"],p["odim"],p["p"],xb,ni,paths.shape[1],
                        True,*flags[1:],has_mask,256,num_warps=4,enable_fp_fusion=True)
        if needed[1]:
            _d2b[(triton.cdiv(e*p["bdim"],16),)](xc,w,g,hxc,hw,hg,src,dst,mask,p["bm_channel"],vc,db,e,
                p["c"],p["xdim"],p["bdim"],p["odim"],p["p"],flags[0],flags[2],flags[3],has_mask,
                triton.next_power_of_2(p["c"]),16,num_warps=4,enable_fp_fusion=True)
        if needed[2] or needed[3]:
            for ni,m,paths in p["gate_groups"]:
                _gate_group[(triton.cdiv(e*paths.shape[1]*p["c"],256),)](xc,b,w,g,hxc,hb,hw,hg,src,dst,mask,paths,vc,
                    dg if needed[3] else g,dwe if needed[2] else g,e,p["c"],p["xdim"],p["bdim"],p["odim"],
                    p["p"],paths.shape[1],ni,m,True,*flags,has_mask,needed[3],needed[2],256,
                    num_warps=4,enable_fp_fusion=True)
        if need_grad:
            if p.get("grouped_forward",False):dy=_group_message_raw(x,b,w,g,src,dst,mask,p,incoming,
                output=cotangent_output,channel_node=xc,channel_tangent=hxc)
            else:
                _d2v[(triton.cdiv(e*p["c"]*p["k"],256),)](x,b,w,g,hx,hb,hw,hg,src,dst,mask,p["km"],dy,
                    e,p["c"],p["xdim"],p["bdim"],p["odim"],p["p"],p["k"],p["imax"],*flags,has_mask,
                    256,num_warps=4,enable_fp_fusion=True)
    dw=dwe.sum(0) if needed[2] else None
    if needed[0] and feature_output is None:dx=_normal_feature_gradient(dx,p)
    return dx,db,dw,dg,dy


class _MessageAdjoint(torch.autograd.Function):
    """The VJP of a four-linear map, with exact mixed second derivatives.

    Each cross derivative is another VJP with one operand replaced; the
    gradient of the upstream node cotangent is a sum of forward maps. This
    avoids constructing edge-sized differentiable projection/scatter graphs.
    """
    @staticmethod
    def forward(ctx,x,b,w,g,src,dst,mask,grad,p,needed):
        ctx.plan=p;ctx.needed=needed;ctx.set_materialize_grads(False)
        grad=grad.contiguous();ctx.save_for_backward(x,b,w,g,src,dst,mask,grad)
        return _vjp_raw(x,b,w,g,src,dst,mask,grad,p,needed)

    @staticmethod
    def backward(ctx,*incoming):
        x,b,w,g,src,dst,mask,grad=ctx.saved_tensors;p=ctx.plan
        originals=(x,b,w,g)
        *needs,need_grad=required_gradients((*originals,grad),(*ctx.needs_input_grad[:4],ctx.needs_input_grad[7]))
        if torch.is_grad_enabled():
            # New descendants isolate partial derivatives even when the saved
            # cotangent depends on an operand. Clones retain higher-order links.
            local=tuple(value.clone() for value in originals);local_grad=grad.clone()
            out=_native(*local,src,dst,mask,p)
            targets=[value for value,need in zip(local,ctx.needed) if need]
            first=torch.autograd.grad(out,targets,local_grad,create_graph=True,allow_unused=True)
            pairs=[(value,cotangent) for value,cotangent in zip(first,[v for v,need in zip(incoming,ctx.needed) if need]) if cotangent is not None and value is not None and value.requires_grad]
            active=[value for value,need in zip((*local,local_grad),(*needs,need_grad)) if need]
            result=iter(torch.autograd.grad(tuple(a for a,_ in pairs),active,tuple(v for _,v in pairs),create_graph=True,allow_unused=True))
            dx,db,dw,dg,dy=[next(result) if need else None for need in (*needs,need_grad)]
        else:
            dx,db,dw,dg,dy=_second_raw(x,b,w,g,src,dst,mask,grad,p,incoming,needs,need_grad)
        return dx,db,dw,dg,None,None,None,dy,None,None


class _Message(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,b,w,g,src,dst,mask,plan):
        ctx.plan=plan;ctx.save_for_backward(x,b,w,g,src,dst,mask)
        return _forward_raw(x,b,w,g,src,dst,mask,plan)

    @staticmethod
    def backward(ctx,grad):
        x,b,w,g,src,dst,mask=ctx.saved_tensors;p=ctx.plan
        needed=required_gradients((x,b,w,g),ctx.needs_input_grad[:4])
        if torch.is_grad_enabled():
            values=_MessageAdjoint.apply(x,b,w,g,src,dst,mask,grad,p,needed)
            return *values,None,None,None,None
        return *_vjp_raw(x,b,w,g,src,dst,mask,grad,p,needed),None,None,None,None


class _PathBlocks(torch.autograd.Function):
    """Avoid full-width slice gradients when users request degree dictionaries."""
    @staticmethod
    def forward(ctx,out,p):
        ctx.set_materialize_grads(False)
        ctx.plan=p;ctx.rows=out.shape[0];ctx.dtype=out.dtype;ctx.device=out.device
        c=p['c'];n=out.shape[0]
        if p.get('inference_degree_layout',False):
            # Physical storage is [degree, node, magnetic, path*channel].
            # Return the ordinary [node, path*channel, magnetic] block views.
            return tuple(out.reshape(-1)[n*p['ooffset'][l]:n*(p['ooffset'][l]+c*p['counts'].get(l,0)*(2*l+1))]
                         .reshape(n,2*l+1,c*p['counts'].get(l,0)).transpose(-2,-1)
                         for l in range(p['lmax']+1))
        return tuple(out[:,p['ooffset'][l]:p['ooffset'][l]+c*p['counts'].get(l,0)*(2*l+1)]
                     .reshape(n,c*p['counts'].get(l,0),2*l+1) for l in range(p['lmax']+1))
    @staticmethod
    def backward(ctx,*gradients):
        if all(v is None for v in gradients):return None,None
        p=ctx.plan;parts=[]
        if p.get('inference_degree_layout',False):
            for l,value in enumerate(gradients):
                width=p['c']*p['counts'].get(l,0)*(2*l+1)
                parts.append(value.transpose(-2,-1).reshape(-1) if value is not None else
                             torch.zeros(ctx.rows*width,dtype=ctx.dtype,device=ctx.device))
            return torch.cat(parts).reshape(ctx.rows,p['odim']),None
        for l,value in enumerate(gradients):
            width=p['c']*p['counts'].get(l,0)*(2*l+1)
            parts.append(value.reshape(ctx.rows,width) if value is not None else
                         torch.zeros((ctx.rows,width),dtype=ctx.dtype,device=ctx.device))
        return torch.cat(parts,-1).contiguous(),None


def aggregate_message(tp,x1,edge_attrs,gates,src,dst,mask,backend="triton",*,generated_options=None):
    """Return node path blocks, or None for the existing reference fallback."""
    sample=next(iter(x1.values()));e=src.numel()
    def option(name,default):
        return (generated_options or {}).get(name,getattr(tp,name,default))
    if (triton is None or not sample.is_cuda or sample.dtype!=torch.float32
            or tp.internal_compute_dtype!=torch.float32 or sample.shape[-2]<1 or tp.num_paths==0
            or any(value.ndim!=3 or value.dtype!=sample.dtype or value.device!=sample.device for value in x1.values())
            or any(value.ndim!=3 or value.shape[-2]!=1 or value.dtype!=sample.dtype or value.device!=sample.device for value in edge_attrs.values())
            or gates.dtype!=sample.dtype or gates.device!=sample.device
            or torch.compiler.is_compiling() or torch.is_autocast_enabled(sample.device.type)
            or torch._C._are_functorch_transforms_active() or get_proxy_mode() is not None or is_fake(sample)
            or (mask is not None and mask.requires_grad)
            or any(int(group["l1"]) not in x1 or int(group["l2"]) not in edge_attrs for group in tp._groups)):
        return None
    plan=_plan(tp,sorted(x1),sample.device,backend in ("triton_projection","triton_csr","triton_remat"));n=sample.shape[0];c=tp.channels
    if backend=="triton_csr":
        from ictc_o3.ops.csr_group_message import live_csr_plan
        plan=live_csr_plan(plan,dst,n)
    elif backend=="triton_remat":
        plan={**plan,"edge_chunk_size":option("_message_chunk_edges",65536),
              "fused_path_forward":getattr(tp,"_fused_path_forward",False),
              "fused_path_adjoint":getattr(tp,"_fused_path_adjoint",False),
              "path_tile_edges":getattr(tp,"_path_tile_edges",1),"path_warps":getattr(tp,"_path_warps",4),
              "geometry_by_column":getattr(tp,"_geometry_by_column",False),"geometry_tile_edges":getattr(tp,"_geometry_tile_edges",32)}
        plan["direct_harmonic"]=getattr(tp,"_direct_harmonic",False)
        plan["int32_index"]=getattr(tp,"_int32_message",False)
    node=torch.cat([x1[l].reshape(n,c*(2*l+1)) for l in sorted(x1)],dim=1).contiguous()
    projectors=tp._get_bond_projectors(sample.device,torch.float32)
    if backend=="triton_remat":
        if any(value.requires_grad for value in projectors):return None
        from ictc_o3.ops.bond_projection import projection_plan
        projection=projection_plan(tp,projectors,sample.device,sample.dtype)
        generated_supported=c>0 and not tp.weight.requires_grad
        plan["generated_forward"]=option("_generated_forward",False) and generated_supported
        plan["generated_adjoint"]=option("_generated_adjoint",False) and generated_supported
        plan['generated_specialized_adjoint']=option('_generated_specialized_adjoint',False) and generated_supported
        plan['generated_csr_adjoint']=option('_generated_csr_adjoint',False) and generated_supported
        plan['generated_csr_fused_adjoint']=option('_generated_csr_fused_adjoint',False) and generated_supported
        plan['generated_topology']=option('_generated_topology','native')
        plan['fused_source_grouping']=option('_fused_source_grouping','degree')
        plan['generated_triton_groups']=option('_generated_triton_groups',False) and generated_supported
        plan['generated_cuda_groups']=option('_generated_cuda_groups',False) and generated_supported
        plan['triton_message_block']=option('_triton_message_block',128)
        plan["generated_second"]=option("_generated_second",False) and generated_supported
        plan['generated_second_roles']=option('_generated_second_roles',False) and generated_supported
        plan['generated_second_channel_tile']=option('_generated_second_channel_tile',None)
        plan['training_adjoint_channel_tile']=option('_training_adjoint_channel_tile',None)
        plan['generated_second_harmonic_split']=option('_generated_second_harmonic_split',False)
        plan['release_packed_inputs']=option('_radial_checkpoint',False)
        plan["generated_layout"]=option("_generated_layout","packed") if generated_supported else "packed"
        plan["generated_blocks"]=option("_generated_blocks",0)
        plan["generated_schedule"]=option("_generated_schedule",0)
        plan["generated_reg_limit"]=option("_generated_reg_limit",0)
        plan["generated_channel_tile"]=option("_generated_channel_tile",0)
        plan["generated_geometry_streaming"]=option("_generated_geometry_streaming",False)
        plan["generated_csr_forward"]=option("_generated_csr_forward",False)
        plan["csr_accumulator_limit"]=option("_csr_accumulator_limit",48)
        plan['csr_slice_search']=option('_csr_slice_search',False)
        plan['csr_neighbor_warps']=option('_csr_neighbor_warps',1)
        magnetic=torch.cat([edge_attrs[l].squeeze(-2) for l in projection["keys"]],-1).contiguous()
    elif backend in ("cuda_projection","triton_projection","triton_csr"):
        from ictc_o3.ops.bond_projection import projected_geometry
        magnetic=projected_geometry(tp,edge_attrs,projectors,sample)
        if magnetic is None:return None
    else:
        banks=[]
        for group,projector in zip(tp._groups,projectors):
            geometry=edge_attrs[int(group["l2"])].squeeze(-2)
            banks.append((geometry@projector).reshape(e,(2*int(group["l1"])+1)*int(group["k_total"])))
        magnetic=torch.cat(banks,dim=1).contiguous()
    gate=gates.reshape(e,tp.num_paths,c).contiguous()
    mask=sample.new_empty(0) if mask is None else mask.to(sample).reshape(e).contiguous()
    if backend=="triton_remat":
        from ictc_o3.ops.rematerialized_message import rematerialized_message
        if option('_generated_compact_cartesian',False):
            raise ValueError('Legacy STF-coordinate flag is retired; reconfigure the model with message_schedule=native or roles')
        out=rematerialized_message(node,magnetic,tp.weight.to(sample).contiguous(),gate,
                                   src.contiguous(),dst.contiguous(),mask,plan,projection)
    elif backend in ("cuda","cuda_projection"):
        from ictc_o3.backends.k100.cuda_message import cuda_message
        out=cuda_message(node,magnetic,tp.weight.to(sample).contiguous(),gate,src.contiguous(),dst.contiguous(),mask,plan)
    elif backend in ("triton","triton_projection","triton_csr"):
        out=_Message.apply(node,magnetic,tp.weight.to(sample).contiguous(),gate,src.contiguous(),dst.contiguous(),mask,plan)
    else:
        raise ValueError("Unsupported message backend")
    return dict(enumerate(_PathBlocks.apply(out,plan)))
