# SPDX-License-Identifier: MIT
"""Live device CSR grouping; integer counters, no cached graph or host scalars.

Sorted input is detected on device and keeps identity edge order. Unsorted
input uses integer atomic placement and does not guarantee stable order within
a node. Geometry/gates stay in original order; no topology is cached.
"""
import torch
from ictc_o3.ops.monomial_operator import triton


if triton is not None:
    import triton.language as tl
    @triton.jit
    def _count(KEYS,COUNTS,SORTED,E,BLOCK:tl.constexpr):
        i=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
        active=i<E
        key=tl.load(KEYS+i,active,0)
        tl.atomic_add(COUNTS+key,1,active,sem='relaxed')
        previous=tl.load(KEYS+tl.maximum(i-1,0),active&(i>0),0)
        bad=tl.sum((active&(i>0)&(key<previous)).to(tl.int32),0)
        if bad>0:tl.atomic_min(SORTED,0,sem='relaxed')

    @triton.jit
    def _place(KEYS,CURSOR,ORDER,SORTED,E,BLOCK:tl.constexpr):
        i=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
        active=i<E
        if tl.load(SORTED)>0:
            tl.store(ORDER+i,i,active)
        else:
            key=tl.load(KEYS+i,active,0)
            position=tl.atomic_add(CURSOR+key,1,active,sem='relaxed')
            tl.store(ORDER+position,i,active)


def topology(keys,n,with_order_hint=False):
    if triton is None:raise RuntimeError('Device CSR requires Triton')
    if not keys.is_cuda or keys.dtype!=torch.int64 or keys.ndim!=1:
        raise ValueError('Device CSR requires int64 CUDA node indices')
    if not isinstance(n,int) or isinstance(n,bool) or n<0:
        raise ValueError('Device CSR requires nonnegative node count')
    keys=keys.contiguous()
    if not keys.numel():
        result=(keys.new_zeros(n+1),keys.new_empty(0))
        return (*result,keys.new_ones(1)) if with_order_hint else result
    if n==0:raise ValueError('A nonempty graph requires nodes')
    # Valid node indices are the same caller contract as the native CSR API.
    counts=keys.new_zeros(n)
    sorted_input=keys.new_ones(1)
    _count[(triton.cdiv(keys.numel(),256),)](keys,counts,sorted_input,keys.numel(),256,num_warps=4)
    ptr=torch.cat((counts.new_zeros(1),counts.cumsum(0)))
    cursor=ptr[:-1].clone()
    order=torch.empty_like(keys)
    _place[(triton.cdiv(keys.numel(),256),)](keys,cursor,order,sorted_input,keys.numel(),256,num_warps=4)
    return (ptr,order,sorted_input) if with_order_hint else (ptr,order)
