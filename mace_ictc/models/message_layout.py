"""One differentiable Triton copy for the existing message-layout boundary.

This keeps the coalesced layout used by graph gather/scatter shaders. It fuses
the per-degree transpose/copy/concat sequence, without caching live features.
Prepared AOTI paths lower the same registered Triton operator into cubins.
"""
from typing import List
from functools import lru_cache
import importlib.util
import itertools
import os
from pathlib import Path
import sys

import torch
from .monomial_operator import triton

_SERIAL=itertools.count()


@lru_cache(maxsize=128)
def _kernel(specs,to_packed):
    # Emit static scalar offsets; Torch 2.7's wrap_triton path does not unwrap
    # constexpr tuple indexing for the installed Triton compiler.
    lines=["import triton", "import triton.language as tl", "@triton.jit",
        "def kernel(X,Y,ELEMENTS,WIDTH:tl.constexpr,BLOCK:tl.constexpr):",
        "    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)",
        "    row=index//WIDTH;column=index%WIDTH",
        "    source=tl.full((BLOCK,),0,tl.int64)"]
    for offset in range(0,len(specs),4):
        base,count,magnetic,channels=specs[offset:offset+4]
        span=magnetic*channels
        lines += [f"    local=column-{base}",
                  f"    path=local//{span};inner=local%{span}"]
        mapped=(f"(inner%{channels})*{magnetic}+inner//{channels}" if to_packed
                else f"(inner%{magnetic})*{channels}+inner//{magnetic}")
        lines += [f"    mapped={mapped}",
                  f"    source=tl.where((local>=0)&(local<{count*span}),row*WIDTH+{base}+path*{span}+mapped,source)"]
    lines += ["    value=tl.load(X+source,index<ELEMENTS,0.)",
              "    tl.store(Y+index,value,index<ELEMENTS)"]
    root=Path(os.environ.get("TRITON_CACHE_DIR",str(Path.home()/".triton/cache")))/"message_layout"
    root.mkdir(parents=True,exist_ok=True)
    name=f"ictc_message_layout_{os.getpid()}_{next(_SERIAL)}"
    path=root/f"{name}.py"
    path.write_text("\n".join(lines)+"\n")
    module_spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(module_spec)
    sys.modules[name]=module
    module_spec.loader.exec_module(module)
    return module.kernel


if triton is not None and hasattr(torch.library, "triton_op"):
    from torch.library import triton_op, wrap_triton

    @triton_op("mace_ictc_layout::message_copy", mutates_args={})
    def convert(x: torch.Tensor, specs: List[int], to_packed: bool) -> torch.Tensor:
        x=x.contiguous()
        if (torch.version.hip and x.is_cuda and x.dtype==torch.float32
                and getattr(torch.cuda.get_device_properties(x.device),'gcnArchName','').split(':')[0]=='gfx926'):
            from .hip_message_layout import copy
            return copy(x,specs,to_packed)
        output = torch.empty_like(x)
        groups = tuple(specs)
        if x.numel():
            wrap_triton(_kernel(groups,to_packed))[(triton.cdiv(x.numel(), 256),)](
                x, output, x.numel(), x.shape[1], 256,
                num_warps=4,
            )
        return output

    def _setup(ctx, inputs, output):
        _, ctx.specs, ctx.to_packed = inputs

    def _backward(ctx, gradient):
        return convert(gradient.contiguous(), ctx.specs, not ctx.to_packed), None, None

    convert.register_autograd(_backward, setup_context=_setup)
else:
    convert = None


def feature_copy(x, plan, *, to_packed):
    specs = [entry for base, magnetic, _ in plan["dx_groups"]
             for entry in (base, 1, magnetic, plan["c"])]
    return convert(x, specs, to_packed)


def output_copy(x, plan, *, to_packed):
    specs = [entry for degree in range(plan["lmax"]+1) if plan["counts"].get(degree, 0)
             for entry in (plan["ooffset"][degree], plan["counts"][degree],
                           2*degree+1, plan["c"])]
    return convert(x, specs, to_packed)
