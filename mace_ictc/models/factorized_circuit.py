"""Generated independent FP32 CG circuits with register-local prefixes.

Only immutable coefficients generate source. Features and all learned weights
are loaded on each invocation. The generated files live in the configured
Triton task cache; no CuEq descriptors, code or kernels are used.
"""
import importlib.util
import os
from pathlib import Path
import sys

import torch
from .adjoint_requirements import required_gradients
from .monomial_operator import triton
from .sparse_weighted_product import _native

_CIRCUIT_KERNELS={}


def _literal(value):
    return repr(float(value))


def _sum(parts):
    return " + ".join(f"({part})" for part in parts) if parts else "0.0"


def _generate(pair, final, size, compact_adjoints=False, species_banked=False):
    width=pair["sizes"][3] if pair is not None else 0
    p=final["sizes"][2];m=final["sizes"][3]
    prefixes={}
    if pair is not None:
        for a,b,_,out,value in pair["entries"]:
            prefixes.setdefault(out,[]).append((a,b,value))
    groups={}
    for left,right,path,out,value in final["entries"]:
        groups.setdefault(left,[]).append((right,path,out,value))
    lines=["import triton", "import triton.language as tl", "", "@triton.jit",
           "def circuit(X,A,G,H,HA,IDX,Y,N,C:tl.constexpr,MODE:tl.constexpr,HASX:tl.constexpr,HASA:tl.constexpr,BLOCK:tl.constexpr):",
           "    row=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)",
           "    n=row//C;c=row%C;valid=row<N*C"]
    def emit(text,indent=1):lines.append("    "*indent+text)
    emit(f"weight_base=(tl.load(IDX+n,valid,0)*C+c)*{p}" if species_banked else f"weight_base=c*{p}")
    for i in range(size):
        emit(f"x{i}=tl.load(X+(n*{size}+{i})*C+c,valid,0.)")
    emit(f"x{size}=tl.full((BLOCK,),1.,tl.float32)")
    emit("if HASX:")
    for i in range(size):emit(f"h{i}=tl.load(H+(n*{size}+{i})*C+c,valid,0.)",2)
    emit(f"h{size}=tl.full((BLOCK,),0.,tl.float32)",2)
    emit("if MODE!=2:")
    for path in range(p):emit(f"w{path}=tl.load(A+weight_base+{path},valid,0.)",2)
    emit("if HASA:")
    for path in range(p):emit(f"a{path}=tl.load(HA+weight_base+{path},valid,0.)",2)
    emit("if MODE==1 or MODE==2:")
    for out in range(m):emit(f"g{out}=tl.load(G+(n*{m}+{out})*C+c,valid,0.)",2)
    emit("if MODE==0 or MODE==3:")
    for out in range(m):emit(f"y{out}=tl.full((BLOCK,),0.,tl.float32)",2)
    emit("elif MODE==1:")
    for i in range(size):emit(f"d{i}=tl.full((BLOCK,),0.,tl.float32)",2)
    emit("else:")
    for path in range(p):emit(f"b{path}=tl.full((BLOCK,),0.,tl.float32)",2)
    for left,entries in groups.items():
        identity=left>=width
        terms=prefixes.get(left,[])
        value=f"x{left-width}" if identity else _sum(f"{_literal(v)}*x{a}*x{b}" for a,b,v in terms)
        tangent=f"h{left-width}" if identity else _sum(f"{_literal(v)}*(h{a}*x{b}+x{a}*h{b})" for a,b,v in terms)
        emit(f"v={value}")
        emit("if HASX:");emit(f"hv={tangent}",2)
        by_output={}
        for right,path,out,coefficient in entries:
            by_output.setdefault(out,[]).append((right,path,coefficient))
        emit("if MODE==0 or MODE==3:")
        for out,rows in by_output.items():
            rhs=_sum(f"{_literal(v)}*w{path}*x{right}" for right,path,v in rows)
            emit(f"t={rhs}",2)
            emit("if MODE==0:",2);emit(f"y{out}=y{out}+v*t",3)
            emit("else:",2)
            emit("if HASX:",3)
            trhs=_sum(f"{_literal(v)}*w{path}*h{right}" for right,path,v in rows)
            emit(f"y{out}=y{out}+hv*t+v*({trhs})",4)
            emit("if HASA:",3)
            arhs=_sum(f"{_literal(v)}*a{path}*x{right}" for right,path,v in rows)
            emit(f"y{out}=y{out}+v*({arhs})",4)
        emit("elif MODE==1:")
        bar=_sum(f"{_literal(v)}*w{path}*x{right}*g{out}" for right,path,out,v in entries)
        emit(f"bv={bar}",2)
        emit("if HASX:",2)
        hbar=_sum(f"{_literal(v)}*w{path}*h{right}*g{out}" for right,path,out,v in entries)
        emit(f"hbv={hbar}",3)
        emit("if HASA:",2)
        abar=_sum(f"{_literal(v)}*a{path}*x{right}*g{out}" for right,path,out,v in entries)
        emit(f"abv={abar}",3)
        if identity:
            i=left-width
            emit("if not HASX and not HASA:",2);emit(f"d{i}=d{i}+bv",3)
            emit("if HASX:",2);emit(f"d{i}=d{i}+hbv",3)
            emit("if HASA:",2);emit(f"d{i}=d{i}+abv",3)
        elif compact_adjoints:
            derivatives={}
            for a,b,coefficient in terms:
                for i,other in ((a,b),(b,a)):
                    derivatives.setdefault(i,[]).append((other,coefficient))
            for i,rows in derivatives.items():
                emit(f"pv="+_sum(f"{_literal(coefficient)}*x{other}" for other,coefficient in rows),2)
                emit("if not HASX and not HASA:",2);emit(f"d{i}=d{i}+bv*pv",3)
                emit("if HASX:",2)
                emit(f"hpv="+_sum(f"{_literal(coefficient)}*h{other}" for other,coefficient in rows),3)
                emit(f"d{i}=d{i}+hbv*pv+bv*hpv",3)
                emit("if HASA:",2);emit(f"d{i}=d{i}+abv*pv",3)
        else:
            for a,b,v in terms:
                for i,other in ((a,b),(b,a)):
                    emit("if not HASX and not HASA:",2)
                    emit(f"d{i}=d{i}+{_literal(v)}*bv*x{other}",3)
                    emit("if HASX:",2)
                    emit(f"d{i}=d{i}+{_literal(v)}*(hbv*x{other}+bv*h{other})",3)
                    emit("if HASA:",2)
                    emit(f"d{i}=d{i}+{_literal(v)}*abv*x{other}",3)
        if compact_adjoints:
            right_rows={}
            for right,path,out,coefficient in entries:
                if right!=size:right_rows.setdefault(right,[]).append((path,out,coefficient))
            for right,rows in right_rows.items():
                rhs=_sum(f"{_literal(coefficient)}*w{path}*g{out}" for path,out,coefficient in rows)
                emit(f"rv={rhs}",2)
                emit("if not HASX and not HASA:",2);emit(f"d{right}=d{right}+v*rv",3)
                emit("if HASX:",2);emit(f"d{right}=d{right}+hv*rv",3)
                emit("if HASA:",2)
                rhs=_sum(f"{_literal(coefficient)}*a{path}*g{out}" for path,out,coefficient in rows)
                emit(f"d{right}=d{right}+v*({rhs})",3)
        else:
            for right,path,out,v in entries:
                if right==size:continue
                multiplier=f"{_literal(v)}*w{path}*g{out}"
                emit("if not HASX and not HASA:",2)
                emit(f"d{right}=d{right}+v*({multiplier})",3)
                emit("if HASX:",2);emit(f"d{right}=d{right}+hv*({multiplier})",3)
                emit("if HASA:",2)
                emit(f"d{right}=d{right}+v*({_literal(v)}*a{path}*g{out})",3)
        emit("else:")
        if compact_adjoints:
            path_rows={}
            for right,path,out,coefficient in entries:path_rows.setdefault(path,[]).append((right,out,coefficient))
            for path,rows in path_rows.items():
                rhs=_sum(f"{_literal(coefficient)}*g{out}*x{right}" for right,out,coefficient in rows)
                emit(f"pv={rhs}",2)
                emit("if HASX:",2)
                hrhs=_sum(f"{_literal(coefficient)}*g{out}*h{right}" for right,out,coefficient in rows)
                emit(f"b{path}=b{path}+hv*pv+v*({hrhs})",3)
                emit("else:",2);emit(f"b{path}=b{path}+v*pv",3)
        else:
            for right,path,out,v in entries:
                emit("if HASX:",2)
                emit(f"b{path}=b{path}+{_literal(v)}*g{out}*(hv*x{right}+v*h{right})",3)
                emit("else:",2)
                emit(f"b{path}=b{path}+{_literal(v)}*g{out}*v*x{right}",3)
    emit("if MODE==0 or MODE==3:")
    for out in range(m):emit(f"tl.store(Y+(n*{m}+{out})*C+c,y{out},valid)",2)
    emit("elif MODE==1:")
    for i in range(size):emit(f"tl.store(Y+(n*{size}+{i})*C+c,d{i},valid)",2)
    emit("else:")
    for path in range(p):emit(f"tl.store(Y+({path}*N+n)*C+c,b{path},valid)",2)
    return "\n".join(lines)+"\n"


def _kernel(plan,species_banked=False):
    signature=plan.get("circuit_signature")
    if signature is None:
        signature=(None if plan["pair"] is None else plan["pair"]["entries"],
                   plan["final"]["entries"],plan["input_size"],plan["final"]["sizes"],
                   bool(plan.get("compact_adjoints",False)))
        plan["circuit_signature"]=signature
    signature=(signature,species_banked)
    if signature in _CIRCUIT_KERNELS:return _CIRCUIT_KERNELS[signature]
    root=Path(os.environ.get("TRITON_CACHE_DIR",str(Path.home()/".triton/cache")))/"independent_circuits"
    root.mkdir(parents=True,exist_ok=True)
    name=f"ictc_circuit_{os.getpid()}_{id(plan)}_{int(species_banked)}"
    path=root/f"{name}.py"
    source=_generate(plan["pair"],plan["final"],plan["input_size"],plan.get("compact_adjoints",False),species_banked)
    path.write_text(source)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    _CIRCUIT_KERNELS[signature]=module.circuit
    return module.circuit


def _raw(x,a,plan,mode=0,g=None,hx=None,ha=None,idx=None):
    if (torch.version.hip and x.is_cuda and x.dtype==torch.float32
            and getattr(torch.cuda.get_device_properties(x.device),'gcnArchName','').split(':')[0]=='gfx926'):
        from .hip_circuit import raw
        return raw(x,a,plan,mode,g,idx,hx=hx,ha=ha)
    n,c,i=x.shape;p=plan["final"]["sizes"][2];m=plan["final"]["sizes"][3]
    packed=x.transpose(1,2).contiguous()
    hp=packed if hx is None else hx.transpose(1,2).contiguous()
    gp=packed if g is None else g.transpose(1,2).contiguous()
    out=x.new_empty((p,n,c) if mode==2 else (n,i if mode==1 else m,c))
    if out.numel():
        compiled=_kernel(plan,idx is not None)[(triton.cdiv(n*c,128),)](packed,a.contiguous(),gp,hp,
            a if ha is None else ha.contiguous(),packed if idx is None else idx,out,n,c,mode,hx is not None,ha is not None,128,
            num_warps=4,enable_fp_fusion=True)
        plan.setdefault("circuit_resources",{})[(mode,hx is not None,ha is not None)]={
            "registers":getattr(compiled,"n_regs",None),"spills":getattr(compiled,"n_spills",None)}
    if mode==2:
        if idx is not None:
            from .species_circuit import reduce_bank
            return reduce_bank(out,idx,a.shape[0])
        return out.sum(1).T.contiguous()
    return out.transpose(1,2).contiguous()


def _reference(x,a,plan):
    pairs=(_native(x,x,x.new_ones(x.shape[1],1),plan["pair"])
           if plan["pair"] is not None else x[:,:,:0])
    return _native(torch.cat((pairs,x),-1),torch.cat((x,x.new_ones(*x.shape[:2],1)),-1),a,plan["final"])


class _Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,a,g,plan,needed):
        ctx.set_materialize_grads(False);ctx.save_for_backward(x,a,g);ctx.plan=plan;ctx.needed=needed
        return (_raw(x,a,plan,1,g) if needed[0] else None,
                _raw(x,a,plan,2,g) if needed[1] else None)

    @staticmethod
    def backward(ctx,hx,ha):
        x,a,g=ctx.saved_tensors;plan=ctx.plan
        needed=required_gradients((x,a,g),ctx.needs_input_grad[:3])
        if torch.is_grad_enabled():
            local=[v.clone() for v in (x,a,g)]
            first_targets=[v for v,need in zip(local[:2],ctx.needed) if need]
            first=torch.autograd.grad(_reference(*local[:2],plan),first_targets,local[2],create_graph=True)
            incoming=[v for v,need in zip((hx,ha),ctx.needed) if need]
            terms=[(d*h).sum() for d,h in zip(first,incoming) if h is not None]
            active=[v for v,need in zip(local,needed) if need]
            derivatives=iter(torch.autograd.grad(sum(terms),active,create_graph=True,allow_unused=True))
            result=[next(derivatives) if need else None for need in needed]
            return *result,None,None
        dx=_raw(x,a,plan,1,g,hx,ha) if needed[0] and (hx is not None or ha is not None) else None
        da=_raw(x,a,plan,2,g,hx) if needed[1] and hx is not None else None
        dg=_raw(x,a,plan,3,None,hx,ha) if needed[2] and (hx is not None or ha is not None) else None
        return dx,da,dg,None,None


class _Circuit(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,a,plan):
        ctx.save_for_backward(x,a);ctx.plan=plan
        return _raw(x,a,plan)

    @staticmethod
    def backward(ctx,g):
        x,a=ctx.saved_tensors;plan=ctx.plan;g=g.contiguous()
        needed=required_gradients((x,a),ctx.needs_input_grad[:2])
        if torch.is_grad_enabled():values=_Adjoint.apply(x,a,g,plan,needed)
        else:values=(_raw(x,a,plan,1,g) if needed[0] else None,_raw(x,a,plan,2,g) if needed[1] else None)
        return *values,None


def factorized_circuit(x,a,plan):
    if triton is not None and x.is_cuda and x.dtype==torch.float32:
        return _Circuit.apply(x,a,plan)
    return _reference(x,a,plan)
