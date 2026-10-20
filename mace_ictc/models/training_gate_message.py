# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Page the radial projection and its first/mixed message adjoints.

Only the narrow radial feature cotangent crosses the first-adjoint boundary.
Wide gate cotangents are recomputed per page for mixed parameter derivatives.
The same live final linear parameters and original Cartesian TP are used.
"""
import torch
import torch.nn.functional as F

from .adjoint_requirements import required_gradients
from .rematerialized_message import _edge_slice, _tiles, _native
from .streamed_gate_message import _StreamedGate


def _gate(h,a,b,p):
    return F.linear(h,a,b if b.numel() else None).reshape(h.shape[0],p['p'],p['c'])


def _expression(x,sh,w,h,a,b,src,dst,mask,p,projection):
    return _native(x,sh,w,_gate(h,a,b,p),src,dst,mask,p,projection)


def _first(x,sh,w,h,a,b,src,dst,mask,v,p,projection,needed,cache):
    from .fused_message import _channel_inputs,_normal_feature_gradient
    from .generated_cartesian_message import generated_adjoint
    nx,ns,nh,na,nb=needed
    dx=torch.zeros_like(x) if nx else None
    ds=torch.empty_like(sh) if ns else None
    dh=torch.empty_like(h) if nh else None
    da=torch.zeros_like(a) if na else None
    db=torch.zeros_like(b) if nb else None
    canonical=p.get('generated_layout')=='canonical'
    xc,vc=(x,v) if canonical else _channel_inputs(x,v,p)
    for start,end in _tiles(src.numel(),p):
        g=cache.get(start)
        if g is None:g=_gate(h[start:end],a,b,p)
        _,dg=generated_adjoint(xc,sh[start:end],w,g,src[start:end],dst[start:end],
            _edge_slice(mask,start,end),vc,dx,p,projection,(nx,ns,nh or na or nb),
            harmonic_output=ds[start:end] if ns else None)
        if dg is not None:
            flat=dg.reshape(end-start,-1)
            if nh:torch.mm(flat,a,out=dh[start:end])
            if na:da.addmm_(flat.t(),h[start:end])
            if nb:db.add_(flat.sum(0))
            del flat
        del g,dg
    if nx and not canonical:dx=_normal_feature_gradient(dx,p)
    return dx,ds,dh,da,db


def _second(x,sh,w,h,a,b,src,dst,mask,v,p,projection,incoming,needed,need_v,cache):
    from .fused_message import _channel_inputs,_channel_node,_normal_feature_gradient,_normal_message_output
    from .generated_cartesian_message import generated_adjoint
    from .generated_message_second import _extension
    second_plan=p if p.get('generated_second_channel_tile') is None else {
        **p,'generated_channel_tile':p['generated_second_channel_tile']}
    tx,ts,th,ta,tb=incoming
    nx,ns,nh,na,nb=needed
    # Coordinate Hessians, canonical layout, and higher derivatives keep the
    # differentiable expression; this fast path serves force-loss parameters.
    assert not ns and p.get('generated_layout')!='canonical'
    dx=torch.zeros_like(x) if nx else None
    dh=torch.zeros_like(h) if nh else None
    da=torch.zeros_like(a) if na else None
    db=torch.zeros_like(b) if nb else None
    dy=x.new_zeros(x.shape[0],p['odim']) if need_v else None
    xc,vc=_channel_inputs(x,v,p)
    txc=_channel_node(tx,p) if tx is not None else xc
    for start,end in _tiles(src.numel(),p):
        hs=h[start:end];ss=sh[start:end]
        g=cache.get(start)
        if g is None:g=_gate(hs,a,b,p)
        tg=None
        if th is not None:tg=F.linear(th[start:end],a)
        if ta is not None:
            extra=F.linear(hs,ta)
            tg=extra if tg is None else tg.add_(extra)
            del extra
        if tb is not None:
            tg=tb.expand(end-start,-1).contiguous() if tg is None else tg.add_(tb)
        if tg is not None:tg=tg.reshape_as(g)
        need_g=nh or na or nb
        dg=torch.empty_like(g) if need_g else None
        if any(value is not None for value in (tx,ts,tg)):
            _extension(second_plan,projection).generated_second(xc,ss.contiguous(),w,g,
                src[start:end],dst[start:end],_edge_slice(mask,start,end),vc,txc,
                ss if ts is None else ts[start:end].contiguous(),g if tg is None else tg,
                xc if dx is None else dx,g if dg is None else dg,vc if dy is None else dy,
                tx is not None,ts is not None,tg is not None,nx,need_g,need_v)
        elif dg is not None:
            dg.zero_()
        if dg is not None:
            flat=dg.reshape(end-start,-1)
            if nh:torch.mm(flat,a,out=dh[start:end])
            if na:da.addmm_(flat.t(),hs)
            if nb:db.add_(flat.sum(0))
            del flat
        del tg,dg
        # Derivatives of the gate tangent th @ A.T + h @ ta.T + tb.
        # They require the original gate cotangent only for these two roles.
        if (nh and ta is not None) or (na and th is not None):
            _,base=generated_adjoint(xc,ss,w,g,src[start:end],dst[start:end],
                _edge_slice(mask,start,end),vc,None,p,projection,(False,False,True))
            flat=base.reshape(end-start,-1)
            if nh and ta is not None:dh[start:end].addmm_(flat,ta)
            if na and th is not None:da.addmm_(flat.t(),th[start:end])
            del flat,base
        del g
    if nx:dx=_normal_feature_gradient(dx,p)
    if need_v:dy=_normal_message_output(dy,p)
    return dx,None,dh,da,db,dy


class _GateAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,sh,w,h,a,b,src,dst,mask,v,p,projection,needed,starts,*gates):
        ctx.set_materialize_grads(False)
        ctx.plan=p;ctx.projection=projection;ctx.needed=needed;ctx.starts=starts
        v=v.contiguous()
        ctx.save_for_backward(x,sh,w,h,a,b,src,dst,mask,v,*gates)
        return _first(x,sh,w,h,a,b,src,dst,mask,v,p,projection,needed,dict(zip(starts,gates)))

    @staticmethod
    def backward(ctx,tx,ts,th,ta,tb):
        x,sh,w,h,a,b,src,dst,mask,v,*gates=ctx.saved_tensors
        p=ctx.plan;projection=ctx.projection
        declared=tuple(ctx.needs_input_grad[i] for i in (0,1,3,4,5,9))
        *needed,need_v=required_gradients((x,sh,h,a,b,v),declared)
        incoming=(tx,ts,th,ta,tb)
        higher=torch.is_grad_enabled()
        if (higher or needed[1] or p.get('generated_layout')=='canonical'
                or not p.get('generated_second',False)):
            with torch.enable_grad():
                local=tuple(value.clone() for value in (x,sh,h,a,b));lv=v.clone()
                lx,ls,lh,la,lb=local
                out=_expression(lx,ls,w,lh,la,lb,src,dst,mask,p,projection)
                first=torch.autograd.grad(out,[value for value,need in zip(local,ctx.needed) if need],
                    lv,create_graph=True,allow_unused=True)
                tangents=[value for value,need in zip(incoming,ctx.needed) if need]
                pairs=[(value,tangent) for value,tangent in zip(first,tangents)
                       if value is not None and value.requires_grad and tangent is not None]
                active=[value for value,need in zip((*local,lv),(*needed,need_v)) if need]
                if pairs and active:
                    grads=iter(torch.autograd.grad(tuple(value for value,_ in pairs),active,
                        tuple(tangent for _,tangent in pairs),create_graph=higher,allow_unused=True))
                    values=tuple(next(grads) if need else None for need in (*needed,need_v))
                else:values=(None,)*6
        else:
            values=_second(x,sh,w,h,a,b,src,dst,mask,v,p,projection,incoming,needed,need_v,
                           dict(zip(ctx.starts,gates)))
        dx,ds,dh,da,db,dv=values
        return dx,ds,None,dh,da,db,None,None,None,dv,None,None,None,None,*([None]*len(gates))


class _TrainingGate(_StreamedGate):
    @staticmethod
    def backward(ctx,grad):
        x,sh,w,h,a,b,src,dst,mask,*gates=ctx.saved_tensors
        needed=required_gradients((x,sh,h,a,b),tuple(ctx.needs_input_grad[i] for i in (0,1,3,4,5)))
        if torch.is_grad_enabled():
            values=_GateAdjoint.apply(x,sh,w,h,a,b,src,dst,mask,grad,ctx.plan,ctx.projection,
                                     needed,ctx.cached_gate_starts,*gates)
        else:
            values=_first(x,sh,w,h,a,b,src,dst,mask,grad.contiguous(),ctx.plan,ctx.projection,
                          needed,dict(zip(ctx.cached_gate_starts,gates)))
        dx,ds,dh,da,db=values
        return dx,ds,None,dh,da,db,None,None,None,None,None
