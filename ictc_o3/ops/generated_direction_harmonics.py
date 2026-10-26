# SPDX-License-Identifier: MIT
"""Optional generated Cartesian harmonics and analytic first/second adjoints.

Uses the actual component-normalized ICTC polynomial constants. No spherical
basis rotation or coefficient threshold is applied. Third derivatives retain
the established differentiable expression.
"""
import importlib.util
import os
from pathlib import Path
import sys

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch._subclasses.fake_tensor import is_fake
from ictc_o3.autograd.adjoint_requirements import required_gradients
from ictc_o3.ops.monomial_operator import triton

_PLANS={}


def _sum(parts):
    parts=list(parts)
    return " + ".join(f"({part})" for part in parts) if parts else "0.0"


def _monomial(exponents):
    return "*".join(f"{axis}{power}" for axis,power in zip("xyz",exponents))


def _differentiate(terms,axis):
    result=[]
    for powers,value in terms:
        if not powers[axis]:continue
        updated=list(powers);updated[axis]-=1
        result.append((tuple(updated),value*powers[axis]))
    return result


def _expression(terms):return _sum(f"{repr(float(value))}*{_monomial(powers)}" for powers,value in terms)


def _plan(n,lmax):
    import ictc_o3.basis.ictd_irreps as basis
    constants=[]
    for l in range(1,lmax+1):
        key=(str(n.device),str(n.dtype),l)
        p=basis._dir_proj_cache_by_dev_dtype.get(key)
        if p is None:p=basis._dir_proj_cpu_f64(l).to(n);basis._dir_proj_cache_by_dev_dtype[key]=p
        coefs=basis._dir_coefs_cache_by_dev_dtype.get(key)
        if coefs is None:coefs=basis._dir_monomial_exps_coefs(l)[1].to(n);basis._dir_coefs_cache_by_dev_dtype[key]=coefs
        scale=basis._dir_component_scale_cache_by_dev_dtype.get(key)
        if scale is None:
            scale=torch.tensor(basis._direction_component_scale_cpu_f64(l),device=n.device,dtype=n.dtype)
            basis._dir_component_scale_cache_by_dev_dtype[key]=scale
        if any(value.requires_grad or value.is_inference() for value in (p,coefs,scale)):return None
        constants.extend((p,coefs,scale))
    key=(str(n.device),lmax,tuple((id(value),value._version) for value in constants))
    if key in _PLANS:return _PLANS[key]
    terms=[[((0,0,0),1.)]]
    for l in range(1,lmax+1):
        p,coefs,scale=constants[(l-1)*3:l*3]
        matrix=p.detach().cpu().double();multiplier=coefs.detach().cpu().double();factor=float(scale)
        for h in range(2*l+1):
            terms.append([(tuple(powers),float(matrix[row,h]*multiplier[row])*factor)
                          for row,powers in enumerate(basis._counts_list(l)) if matrix[row,h]!=0])
    hd=(lmax+1)**2
    lines=["import triton", "import triton.language as tl", "", "@triton.jit",
           "def kernel(N,G,H,Y,DN,DG,E,MODE:tl.constexpr,NEEDN:tl.constexpr,NEEDG:tl.constexpr,BLOCK:tl.constexpr):",
           "    e=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)","    valid=e<E"]
    def emit(value,indent=1):lines.append("    "*indent+value)
    for index,axis in enumerate("xyz"):
        emit(f"{axis}0=tl.full((BLOCK,),1.,tl.float32)")
        emit(f"{axis}1=tl.load(N+e*3+{index},valid,0.)")
        for power in range(2,lmax+1):emit(f"{axis}{power}={axis}{power-1}*{axis}1")
    emit("if MODE!=0:")
    for axis in range(3):emit(f"d{axis}=tl.full((BLOCK,),0.,tl.float32)",2)
    emit("if MODE==2:")
    for axis in range(3):emit(f"h{axis}=tl.load(H+e*3+{axis},valid,0.)",2)
    for h,rows in enumerate(terms):
        first=[_differentiate(rows,axis) for axis in range(3)]
        emit("if MODE==0:")
        emit(f"tl.store(Y+e*{hd}+{h},{_expression(rows)},valid)",2)
        emit("else:")
        emit(f"g=tl.load(G+e*{hd}+{h},valid,0.)",2)
        for axis in range(3):emit(f"j{axis}={_expression(first[axis])}",2)
        emit("if MODE==1:",2)
        for axis in range(3):emit(f"d{axis}=d{axis}+g*j{axis}",3)
        emit("else:",2)
        emit("if NEEDG:",3)
        emit(f"tl.store(DG+e*{hd}+{h},h0*j0+h1*j1+h2*j2,valid)",4)
        emit("if NEEDN:",3)
        for axis in range(3):
            value=_sum(f"h{other}*({_expression(_differentiate(first[axis],other))})" for other in range(3))
            emit(f"d{axis}=d{axis}+g*({value})",4)
    emit("if MODE!=0:")
    for axis in range(3):emit(f"tl.store(DN+e*3+{axis},d{axis},valid)",2)
    root=Path(os.environ.get("TRITON_CACHE_DIR",str(Path.home()/".triton/cache")))/"cartesian_direction"
    root.mkdir(parents=True,exist_ok=True)
    name=f"ictc_direction_{os.getpid()}_{len(_PLANS)}";path=root/f"{name}.py"
    path.write_text("\n".join(lines)+"\n")
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    result={"kernel":module.kernel,"lmax":lmax,"hdim":hd,"constants":tuple(constants),"terms":terms};_PLANS[key]=result
    return result


def _raw(n,p,mode=0,g=None,h=None,needed=(True,True)):
    if (torch.version.hip and p['lmax']<=3
            and getattr(torch.cuda.get_device_properties(n.device),'gcnArchName','').split(':')[0]=='gfx926'):
        from ictc_o3.backends.k100.hip_direction_harmonics import raw
        return raw(n,p,mode,g,h,needed)
    n=n.contiguous();e=n.shape[0]
    y=n.new_empty(e,p["hdim"]) if mode==0 else n
    dn=torch.empty_like(n) if mode!=0 else n
    dg=n.new_empty(e,p["hdim"]) if mode==2 and needed[1] else n
    if e:p["kernel"][(triton.cdiv(e,256),)](n,n if g is None else g.contiguous(),n if h is None else h.contiguous(),
        y,dn,dg,e,mode,*needed,256,num_warps=4,enable_fp_fusion=True)
    if mode==0:return y
    if mode==1:return dn
    return dn if needed[0] else None,dg if needed[1] else None


def _native(n,p):
    from ictc_o3.basis.ictd_irreps import direction_harmonics_all
    return torch.cat(direction_harmonics_all(n,p["lmax"]),-1)


class _Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx,n,g,p):
        ctx.plan=p;ctx.save_for_backward(n,g,*p["constants"])
        return _raw(n,p,1,g)

    @staticmethod
    def backward(ctx,h):
        n,g,*_=ctx.saved_tensors;p=ctx.plan
        needed=required_gradients((n,g),ctx.needs_input_grad[:2])
        if torch.is_grad_enabled():
            ln=n.clone();lg=g.clone();value=_native(ln,p)
            first=torch.autograd.grad(value,ln,lg,create_graph=True)[0]
            inputs=[tensor for tensor,need in zip((ln,lg),needed) if need]
            values=iter(torch.autograd.grad(first,inputs,h,create_graph=True,allow_unused=True))
            results=tuple(next(values) if need else None for need in needed)
        else:results=_raw(n,p,2,g,h,needed)
        return *results,None


class _Harmonics(torch.autograd.Function):
    @staticmethod
    def forward(ctx,n,p):
        ctx.plan=p;ctx.save_for_backward(n,*p["constants"])
        return _raw(n,p)

    @staticmethod
    def backward(ctx,g):
        n,*_=ctx.saved_tensors;p=ctx.plan
        return (_Adjoint.apply(n,g,p) if torch.is_grad_enabled() else _raw(n,p,1,g)),None


def generated_direction_harmonics(n,lmax):
    if (torch.compiler.is_compiling() or get_proxy_mode() is not None or is_fake(n)
            or torch._C._are_functorch_transforms_active()):return None
    if (triton is None or not n.is_cuda or n.dtype!=torch.float32 or n.ndim!=2 or n.shape[-1]!=3
            or not isinstance(lmax,int) or isinstance(lmax,bool) or lmax<0 or torch.is_autocast_enabled(n.device.type)):
        return None
    p=_plan(n,int(lmax))
    if p is None:return None
    output=_Harmonics.apply(n,p)
    return list(output.split([2*l+1 for l in range(lmax+1)],-1))
