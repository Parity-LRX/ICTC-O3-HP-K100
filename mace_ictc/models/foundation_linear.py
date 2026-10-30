# SPDX-License-Identifier: MIT
"""Live e3nn linear maps in contiguous node/component rows.

Only instruction metadata is planned. Effective matrices are rebuilt from the
canonical parameter views in the current graph, including mixed derivatives.
No learned weights or features are cached.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def linear_plan(linear):
    if not linear.internal_weights or not linear.shared_weights or linear.weight.ndim != 1:
        raise ValueError("Foundation linears require shared, internal one-dimensional weights")
    if linear.bias.numel():
        raise ValueError("Foundation node/component linears currently require bias-free maps")
    inputs = tuple((int(mul), int(ir.dim), sl) for (mul, ir), sl in
                   zip(linear.irreps_in, linear.irreps_in.slices()))
    outputs = []
    offset = 0
    instructions = []
    for ins in linear.instructions:
        if ins.i_in < 0:
            raise ValueError("Bias instructions require the native linear route")
        size = math.prod(ins.path_shape)
        instructions.append((ins.i_in, ins.i_out, slice(offset, offset + size),
                             tuple(ins.path_shape), float(ins.path_weight)))
        offset += size
    if offset != linear.weight.numel():
        raise ValueError("Linear instruction shapes do not cover the canonical weights")
    for index, (mul, ir) in enumerate(linear.irreps_out):
        paths = tuple(value for value in instructions if value[1] == index)
        outputs.append((int(mul), int(ir.dim), paths))
    return inputs, tuple(outputs)


def node_component_linear(features, linear, plan):
    """Apply the original e3nn instruction weights as channel GEMMs."""
    inputs, outputs = plan
    shape = features.shape[:-1]
    flat = features.reshape(-1, features.shape[-1])
    result = []
    for cout, angular, paths in outputs:
        if not paths:
            result.append(flat.new_zeros(flat.shape[0], cout * angular))
            continue
        blocks, weights = [], []
        for i_in, _, sl, weight_shape, scale in paths:
            cin, width, feature_slice = inputs[i_in]
            if width != angular:
                raise ValueError("Linear instructions must preserve angular degree")
            blocks.append(flat[:, feature_slice].reshape(-1, cin, width).transpose(1, 2))
            weights.append(linear.weight[sl].reshape(weight_shape) * scale)
        rows = (blocks[0] if len(blocks) == 1 else torch.cat(blocks, -1)).reshape(-1, sum(w.shape[0] for w in weights))
        weight = weights[0] if len(weights) == 1 else torch.cat(weights, 0)
        value = F.linear(rows, weight.T).reshape(flat.shape[0], angular, cout)
        result.append(value.transpose(1, 2).reshape(flat.shape[0], cout * angular))
    return torch.cat(result, -1).reshape(*shape, linear.irreps_out.dim)


def projected_species_input(radial, source, target, source_index, target_index, linear):
    """Keep the canonical affine values and factor its live derivative graph.

    Reordering this FP32 affine into three projections changes force rounding.
    A force loss can amplify those small output differences in its parameter
    gradient, despite an accurate mixed derivative at a common cotangent.

    Evaluate the original GEMM without retaining its wide per-edge input. The
    equivalent factored expression supplies all first and mixed derivatives,
    including derivatives of the current species tables and canonical weights.
    No learned value is cached. The zero-valued term below preserves the
    canonical forward bits without retaining the original GEMM's adjoint graph.
    """
    nr, ns, nt = radial.shape[-1], source.shape[-1], target.shape[-1]
    if linear.in_features != nr + ns + nt:
        raise ValueError("Radial/species widths do not match the original first linear")
    weight = linear.weight
    with torch.no_grad():
        canonical = F.linear(torch.cat((radial, F.embedding(source_index, source),
            F.embedding(target_index, target)), -1), weight, linear.bias)
    if not torch.is_grad_enabled():
        return canonical
    value = F.linear(radial, weight[:, :nr], linear.bias)
    left = F.linear(source, weight[:, nr:nr + ns])
    right = F.linear(target, weight[:, nr + ns:])
    factored = value + F.embedding(source_index, left) + F.embedding(target_index, right)
    return canonical + (factored - factored.detach())
