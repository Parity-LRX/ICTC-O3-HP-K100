# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Bounded high-degree SC kernels for frozen, dynamic-shape AOTI inference.

The exact prefix and adjoint tables come from the qualified eager plan. Species
weights are indexed in the kernel, without a node-sized projected weight bank.
Only the tables used by the force graph are captured; dense reference U stays
registered on the source model for checkpoints and reference fallback.
"""
from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _map(L, R, W, S, G, IDS, COEF, Y, N, C: tl.constexpr,
         A: tl.constexpr, B: tl.constexpr, P: tl.constexpr, M: tl.constexpr,
         O: tl.constexpr, K: tl.constexpr, MODE: tl.constexpr,
         SPECIES: tl.constexpr, BO: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * BO + tl.arange(0, BO)
    f = row % O
    c = (row // O) % C
    n = row // (C * O)
    valid = row < N * C * O
    k = tl.arange(0, BK)
    active = valid[:, None] & (k[None, :] < K)
    a = tl.load(IDS + f[:, None] * K + k[None, :], active, 0)
    b = tl.load(IDS + O * K + f[:, None] * K + k[None, :], active, 0)
    p = tl.load(IDS + 2 * O * K + f[:, None] * K + k[None, :], active, 0)
    q = tl.load(IDS + 3 * O * K + f[:, None] * K + k[None, :], active, 0)
    coefficient = tl.load(COEF + f[:, None] * K + k[None, :], active, 0.)
    z = tl.load(S + n, valid, 0) if SPECIES else tl.full((BO,), 0, tl.int64)
    w = tl.load(W + (z[:, None] * C + c[:, None]) * P + p, active, 0.)
    if MODE == 0:
        l = tl.load(L + (n[:, None] * C + c[:, None]) * A + a, active, 0.)
        r = tl.load(R + (n[:, None] * C + c[:, None]) * B + b, active, 0.)
        value = (l * r) * (coefficient * w)
    elif MODE == 1:
        r = tl.load(R + (n[:, None] * C + c[:, None]) * B + b, active, 0.)
        g = tl.load(G + (n[:, None] * C + c[:, None]) * M + q, active, 0.)
        value = (r * g) * (coefficient * w)
    else:
        l = tl.load(L + (n[:, None] * C + c[:, None]) * A + a, active, 0.)
        g = tl.load(G + (n[:, None] * C + c[:, None]) * M + q, active, 0.)
        value = (l * g) * (coefficient * w)
    tl.store(Y + row, tl.sum(value, 1), valid)


def prepare_sparse_contraction(owner, plan, name):
    """Return an exportable evaluator over already verified GPU constants."""
    c = int(owner.channels)
    size = plan["input_size"]

    def tables(stem, leg, modes):
        result = []
        for mode in modes:
            ids, coeff = leg["reductions"][mode]
            names = (f"_deployment_{stem}_ids_{mode}", f"_deployment_{stem}_coeff_{mode}")
            for key, value in zip(names, (ids, coeff)):
                owner.register_buffer(key, value, persistent=False)
            result.extend(names)
        return tuple(result)

    def load(keys):
        return tuple(getattr(owner, key) for key in keys)

    def launch(l, r, w, s, g, ids, coeff, sizes, mode, species):
        a, b, p, m = sizes
        o = (m, a, b)[mode]
        k = ids.shape[-1]
        bk = triton.next_power_of_2(k)
        bo = max(1, min(16, 2048 // bk))
        warps = 8 if bk > 2048 else 4
        n = l.shape[0]
        out = l.new_empty(n, c, o)
        wrap_triton(_map)[(triton.cdiv(n * c * o, bo),)](
            l, r, w, s, g, ids, coeff, out, n, c, a, b, p, m,
            o, k, mode, species, bo, bk, num_warps=warps, enable_fp_fusion=True)
        return out

    pair_keys = ()
    if plan["pair"] is not None:
        pair_keys = tables("pair", plan["pair"], (0,)) + tables("pair_adjoint", plan["pair_adjoint"], (1,))
        owner.register_buffer("_deployment_pair_ones", plan["ones"], persistent=False)
        pair_sizes = plan["pair"]["sizes"]

        @triton_op(name("quadratic_backward"), mutates_args={})
        def pair_backward(x: torch.Tensor, w: torch.Tensor, ids: torch.Tensor,
                          coeff: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            return launch(x, x, w, ids, g.contiguous(), ids, coeff, pair_sizes, 1, False)

        @triton_op(name("quadratic"), mutates_args={})
        def pair_forward(x: torch.Tensor, w: torch.Tensor, ids: torch.Tensor, coeff: torch.Tensor,
                         adj_ids: torch.Tensor, adj_coeff: torch.Tensor) -> torch.Tensor:
            return launch(x, x, w, ids, x, ids, coeff, pair_sizes, 0, False)

        def pair_setup(ctx, inputs, output):
            x, w, _, _, adj_ids, adj_coeff = inputs
            ctx.save_for_backward(x, w, adj_ids, adj_coeff)

        def pair_grad(ctx, g):
            return pair_backward(*ctx.saved_tensors, g), None, None, None, None, None

        pair_forward.register_autograd(pair_grad, setup_context=pair_setup)

    final_keys = tables("final", plan["final"], (0, 1, 2))
    final_sizes = plan["final"]["sizes"]

    @triton_op(name("sparse_contraction_backward"), mutates_args={})
    def final_backward(l: torch.Tensor, r: torch.Tensor, w: torch.Tensor, s: torch.Tensor,
                       lids: torch.Tensor, lcoeff: torch.Tensor, rids: torch.Tensor,
                       rcoeff: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g = g.contiguous()
        return (launch(l, r, w, s, g, lids, lcoeff, final_sizes, 1, True),
                launch(l, r, w, s, g, rids, rcoeff, final_sizes, 2, True))

    @triton_op(name("sparse_contraction"), mutates_args={})
    def final_forward(l: torch.Tensor, r: torch.Tensor, w: torch.Tensor, s: torch.Tensor,
                      ids: torch.Tensor, coeff: torch.Tensor, lids: torch.Tensor, lcoeff: torch.Tensor,
                      rids: torch.Tensor, rcoeff: torch.Tensor) -> torch.Tensor:
        return launch(l, r, w, s, l, ids, coeff, final_sizes, 0, True)

    def final_setup(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:4], *inputs[6:])

    def final_grad(ctx, g):
        dl, dr = final_backward(*ctx.saved_tensors, g)
        return dl, dr, None, None, None, None, None, None, None, None

    final_forward.register_autograd(final_grad, setup_context=final_setup)

    def evaluate(x, idx):
        native = (x @ owner._deployment_q).contiguous()
        pairs = (pair_forward(native, owner._deployment_pair_ones, *load(pair_keys))
                 if pair_keys else native[:, :, :0])
        left = torch.cat((pairs, native), -1)
        right = torch.cat((native, native.new_ones(native.shape[0], c, 1)), -1)
        value = final_forward(left, right, owner._deployment_bank, idx.contiguous(), *load(final_keys))
        return torch.cat([(part @ getattr(owner, f"_deployment_qout_{l}").T).reshape(x.shape[0], -1)
                          for l, part in enumerate(value.split(plan["sizes"], -1))], -1)

    return evaluate
