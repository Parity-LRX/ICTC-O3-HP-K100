"""Parallel neighbor reduction within live destination segments, in FP32."""
from bisect import bisect_left,bisect_right
import torch
from .monomial_operator import triton
if triton is not None:
    import triton.language as tl

    @triton.jit
    def _destination_segment(X,B,W,G,S,M,RP,O,E,
        C:tl.constexpr,XDIM:tl.constexpr,BDIM:tl.constexpr,ODIM:tl.constexpr,P:tl.constexpr,
        PATH:tl.constexpr,XB:tl.constexpr,BB:tl.constexpr,BS:tl.constexpr,OB:tl.constexpr,
        I:tl.constexpr,OM:tl.constexpr,NODESTART:tl.constexpr,EDGESTART:tl.constexpr,
        EDGEEND:tl.constexpr,MASK:tl.constexpr,BN:tl.constexpr,BC:tl.constexpr):
        node=tl.program_id(0).to(tl.int64)+NODESTART;mag=tl.program_id(1)
        begin=tl.maximum(tl.load(RP+node),EDGESTART);end=tl.minimum(tl.load(RP+node+1),EDGEEND)
        r=tl.arange(0,BN);c=tl.arange(0,BC)
        total=tl.full((BN,BC),0,tl.float32)
        w=tl.load(W+PATH*C+c,c<C,0.)
        for offset in range(tl.cdiv(tl.maximum(end-begin,0),BN)):
            global_edge=begin+offset*BN+r;edge=global_edge-EDGESTART
            active=(global_edge[:,None]<end)&(c[None,:]<C)
            src=tl.load(S+edge,global_edge<end,0)
            y=tl.full((BN,BC),0,tl.float32)
            for i in tl.static_range(I):
                x=tl.load(X+src[:,None]*XDIM+XB+i*C+c[None,:],active,0.)
                b=tl.load(B+edge*BDIM+BB+i*BS+mag,global_edge<end,0.)
                y=y+x*b[:,None]
            g=tl.load(G+(edge[:,None]*P+PATH)*C+c[None,:],active,0.)
            value=(y*w[None,:])*g
            if MASK:value=value*tl.load(M+edge,global_edge<end,0.)[:,None]
            total=total+value
        value=tl.sum(total,0)
        tl.atomic_add(O+node*ODIM+OB+mag*C+c,value,c<C,sem="relaxed")


def destination_plan(dst,n):
    counts=torch.bincount(dst,minlength=n)
    ptr=torch.cat((counts.new_zeros(1),counts.cumsum(0)))
    return ptr,ptr.cpu().tolist()


def segmented_forward(x,b,w,g,src,mask,out,p,ptr,host_ptr,start,end):
    first=min(len(host_ptr)-2,max(0,bisect_right(host_ptr,start)-1))
    last=min(len(host_ptr)-1,bisect_left(host_ptr,end))
    if last<=first:return
    for path,ni,xb,bb,bs,ob,om in p["path_specs"]:
        _destination_segment[(last-first,om)](x,b,w,g,src,mask,ptr,out,end-start,p["c"],p["xdim"],p["bdim"],
            p["odim"],p["p"],path,xb,bb,bs,ob,ni,om,first,start,end,bool(mask.numel()),
            p.get("segment_neighbors",32),triton.next_power_of_2(p["c"]),num_warps=4,enable_fp_fusion=True)
