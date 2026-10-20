"""Symmetric monomial GEMMs without node-sized high-rank coefficients.

Permutations of feature indices give the same monomial. Their constant U
coefficients are summed without a coefficient threshold. Learned
species/channel coefficients are rebuilt every invocation.
"""
import itertools

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch._subclasses.fake_tensor import is_fake
from .monomial_operator import monomial_plan
from .polynomial_gemm import polynomial_gemm


def _constant_plan(module, order, x):
    u = module.U_tensors(order)
    key = (id(u), u._version, x.device, x.dtype, order)
    cache = getattr(module, "_compact_u_cache", None)
    if cache is None:
        cache = {}; module._compact_u_cache = cache
    if order in cache and cache[order][0] == key:
        return cache[order][1:]
    i = x.shape[-1]
    tuples = list(itertools.combinations_with_replacement(range(i), order))
    lookup = {indices: t for t, indices in enumerate(tuples)}
    mapping = torch.tensor([lookup[tuple(sorted(indices))]
                            for indices in itertools.product(range(i), repeat=order)],
                           device=x.device, dtype=torch.long)
    with torch.no_grad():
        # FP64 short permutation sums reduce planning roundoff before the
        # immutable constant is cast back to the execution dtype.
        value = u.to(device=x.device,dtype=torch.float64 if x.dtype==torch.float32 else x.dtype)
        if module.output_lmax == 0:
            value = value.unsqueeze(0)
        compressed = value.new_zeros(value.shape[0], len(tuples), value.shape[-1])
        compressed.index_add_(1, mapping, value.reshape(value.shape[0], i**order, value.shape[-1]))
        compressed = compressed.to(x)
        indices = torch.tensor(tuples, device=x.device, dtype=torch.long).T.contiguous()
        derivative_plan = monomial_plan(tuples, i, x.device)
    cache[order] = (key, compressed, indices, derivative_plan)
    return compressed, indices, derivative_plan


def _shared_plan(contractions,order,x):
    key=(tuple((id(module.U_tensors(order)),module.U_tensors(order)._version) for module in contractions),x.device,x.dtype)
    owner=contractions[0]
    cache=getattr(owner,"_compact_common_cache",None)
    if cache is None:cache={};owner._compact_common_cache=cache
    if order in cache and cache[order][0]==key:return cache[order][1:]
    constants=[_constant_plan(module,order,x) for module in contractions]
    with torch.no_grad():
        # Keep every nonzero coefficient, however small. Only identically
        # absent monomials across all output irreps can be removed.
        active=torch.stack([(u!=0).any(0).any(-1) for u,_,_ in constants]).any(0)
        rows=torch.nonzero(active,as_tuple=True)[0]
        indices=constants[0][1].index_select(1,rows).contiguous()
        pieces=[u.index_select(1,rows) for u,_,_ in constants]
        tuples=[tuple(row) for row in indices.T.tolist()]
        derivative_plan=monomial_plan(tuples,x.shape[-1],x.device)
    cache[order]=(key,pieces,indices,derivative_plan)
    return pieces,indices,derivative_plan


def compact_contractions(contractions, x, idx):
    """Evaluate all output irreps with shared monomials; None means fallback."""
    if (not contractions or torch.compiler.is_compiling() or get_proxy_mode() is not None
            or torch._C._are_functorch_transforms_active() or is_fake(x)):
        return None
    correlation = contractions[0].correlation
    if correlation > 3 or any(module.correlation != correlation for module in contractions):
        return None
    n, c, _ = x.shape
    sizes = [2*module.output_lmax+1 for module in contractions]
    m = sum(sizes)
    idx = idx.to(device=x.device, dtype=torch.long)
    banks = {}; indices = {}; derivative_plans = {}
    for order in range(1, correlation+1):
        pieces = []
        constants,ids,derivative_plan = _shared_plan(contractions,order,x)
        for module,u in zip(contractions,constants):
            weights = [module.weights_max, *module.weights]
            weight = weights[correlation-order].to(x)
            pieces.append(torch.einsum("mtp,zpc->zctm", u, weight))
        banks[order] = torch.cat(pieces, dim=-1)
        indices[order] = ids
        derivative_plans[order] = derivative_plan
    output = x.new_zeros(n, c, m)
    for species in range(contractions[0].weights_max.shape[0]):
        rows = torch.nonzero(idx == species, as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        features = x.index_select(0, rows)
        value = features.new_zeros(c, rows.numel(), m)
        for order in range(1, correlation+1):
            value = value + polynomial_gemm(features, banks[order][species], indices[order], derivative_plans[order])
        output = output.index_add(0, rows, value.transpose(0, 1))
    return torch.cat([part.reshape(n, c*size) for part, size in zip(output.split(sizes, -1), sizes)], -1)
