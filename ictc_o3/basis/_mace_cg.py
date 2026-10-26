###########################################################################################
# Higher Order Real Clebsch Gordan (based on e3nn by Mario Geiger)
# Authors: Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
#
# Local minimal copy for spherical-fix native MACE symmetric contraction.
###########################################################################################

from __future__ import annotations

import collections
import itertools
import os
from typing import Iterator, List, Union

import numpy as np
import torch
from e3nn import o3

try:
    import cuequivariance as cue

    CUET_AVAILABLE = True
except ImportError:
    cue = None
    CUET_AVAILABLE = False

USE_CUEQ_CG = os.environ.get("MACE_USE_CUEQ_CG", "0").lower() in (
    "1",
    "true",
    "yes",
    "y",
)

_TP = collections.namedtuple("_TP", "op, args")
_INPUT = collections.namedtuple("_INPUT", "tensor, start, stop")


def _wigner_nj(
    irrepss: List[o3.Irreps],
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
    filter_ir_out=None,
):
    irrepss = [o3.Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]
    if filter_ir_out is not None:
        filter_ir_out = [o3.Irrep(ir) for ir in filter_ir_out]

    if len(irrepss) == 1:
        (irreps,) = irrepss
        ret = []
        e = torch.eye(irreps.dim, dtype=dtype)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.dim)
                ret += [(ir, _INPUT(0, sl.start, sl.stop), e[sl])]
                i += ir.dim
        return ret if filter_ir_out is None else [row for row in ret if row[0] in filter_ir_out]

    *irrepss_left, irreps_right = irrepss
    ret = []
    for ir_left, path_left, c_left in _wigner_nj(
        irrepss_left,
        normalization=normalization,
        filter_ir_mid=filter_ir_mid,
        dtype=dtype,
    ):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue
                if filter_ir_out is not None and ir_out not in filter_ir_out:
                    continue

                c = o3.wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype)
                if normalization == "component":
                    c *= ir_out.dim**0.5
                if normalization == "norm":
                    c *= ir_left.dim**0.5 * ir.dim**0.5

                c = torch.einsum("jk,ijl->ikl", c_left.flatten(1), c)
                c = c.reshape(
                    ir_out.dim,
                    *(irreps.dim for irreps in irrepss_left),
                    ir.dim,
                )
                for u in range(mul):
                    e = torch.zeros(
                        ir_out.dim,
                        *(irreps.dim for irreps in irrepss_left),
                        irreps_right.dim,
                        dtype=dtype,
                    )
                    sl = slice(i + u * ir.dim, i + (u + 1) * ir.dim)
                    e[..., sl] = c
                    ret += [
                        (
                            ir_out,
                            _TP(
                                op=(ir_left, ir, ir_out),
                                args=(
                                    path_left,
                                    _INPUT(len(irrepss_left), sl.start, sl.stop),
                                ),
                            ),
                            e,
                        )
                    ]
            i += mul * ir.dim
    return sorted(ret, key=lambda x: x[0])


def U_matrix_real(
    irreps_in: Union[str, o3.Irreps],
    irreps_out: Union[str, o3.Irreps],
    correlation: int,
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
    use_cueq_cg: bool = False,
    use_nonsymmetric_product: bool = False,
):
    irreps_out = o3.Irreps(irreps_out)
    irrepss = [o3.Irreps(irreps_in)] * int(correlation)

    if use_cueq_cg is None:
        use_cueq_cg = USE_CUEQ_CG
    if correlation == 4 and not use_cueq_cg:
        filter_ir_mid = [(i, 1 if i % 2 == 0 else -1) for i in range(12)]
    if use_cueq_cg and CUET_AVAILABLE:
        return compute_U_cueq(
            irreps_in,
            irreps_out=irreps_out,
            correlation=int(correlation),
            use_nonsymmetric_product=use_nonsymmetric_product,
            dtype=dtype,
        )

    try:
        wigners = _wigner_nj(irrepss, normalization, filter_ir_mid, dtype,
                            filter_ir_out=[ir for _,ir in irreps_out])
    except NotImplementedError as exc:
        if use_cueq_cg and CUET_AVAILABLE:
            return compute_U_cueq(
                irreps_in,
                irreps_out=irreps_out,
                correlation=int(correlation),
                use_nonsymmetric_product=use_nonsymmetric_product,
                dtype=dtype,
            )
        raise NotImplementedError(
            "The requested native Clebsch-Gordan coefficients are unavailable. "
            "No cuEq basis is substituted when use_cueq_cg=False."
        ) from exc
    current_ir = wigners[0][0] if wigners else None
    out = []
    stack = torch.tensor([], dtype=dtype)

    for ir, _, base_o3 in wigners:
        # Only the scalar output axis disappears. Unqualified squeeze also
        # removes singleton input axes for L=0, destroying polynomial order.
        base_o3 = base_o3.squeeze(0) if ir.l == 0 else base_o3
        if ir in irreps_out and ir == current_ir:
            stack = torch.cat((stack, base_o3.unsqueeze(-1)), dim=-1)
            last_ir = current_ir
        elif ir in irreps_out and ir != current_ir:
            if len(stack) != 0:
                out += [last_ir, stack]
            stack = base_o3.unsqueeze(-1)
            current_ir, last_ir = ir, ir
        else:
            current_ir = ir

    try:
        out += [last_ir, stack]
    except UnboundLocalError:
        first_dim = irreps_out.dim
        if first_dim != 1:
            size = [first_dim] + [o3.Irreps(irreps_in).dim] * correlation + [1]
        else:
            size = [o3.Irreps(irreps_in).dim] * correlation + [1]
        out = [str(irreps_out)[:-2], torch.zeros(size, dtype=dtype)]
    return out


if CUET_AVAILABLE:

    def compute_U_cueq(
        irreps_in,
        irreps_out,
        correlation=2,
        use_nonsymmetric_product=False,
        dtype=None,
    ):
        if dtype is None:
            dtype = torch.get_default_dtype()
        U = []
        irreps_in_cue = cue.Irreps(O3_e3nn, str(irreps_in))
        irreps_out_cue = cue.Irreps(O3_e3nn, str(irreps_out))
        for _, ir in irreps_out_cue:
            try:
                U_matrix_full_symm = cue.reduced_symmetric_tensor_product_basis(
                    irreps_in_cue,
                    correlation,
                    keep_ir=ir,
                    layout=cue.ir_mul,
                ).array
                if use_nonsymmetric_product:
                    try:
                        U_matrix_full_antisymmetric = cue.reduced_antisymmetric_tensor_product_basis(
                            irreps_in_cue,
                            correlation,
                            keep_ir=ir,
                            layout=cue.ir_mul,
                        ).array
                        U_matrix_full = torch.cat(
                            (U_matrix_full_symm, U_matrix_full_antisymmetric),
                            dim=-1,
                        )
                    except ValueError:
                        continue
                else:
                    U_matrix_full = U_matrix_full_symm
            except ValueError:
                if ir.dim == 1:
                    out_shape = (*([irreps_in_cue.dim] * correlation), 1)
                else:
                    out_shape = (ir.dim, *([irreps_in_cue.dim] * correlation), 1)
                return [torch.zeros(out_shape, dtype=torch.get_default_dtype())]

            if U_matrix_full.shape[-1] == 0:
                if ir.dim == 1:
                    out_shape = (*([irreps_in_cue.dim] * correlation), 1)
                else:
                    out_shape = (ir.dim, *([irreps_in_cue.dim] * correlation), 1)
                return [torch.zeros(out_shape, dtype=torch.get_default_dtype())]

            U.append(str(ir))
            U_matrix_full = torch.tensor(
                U_matrix_full.reshape(*([irreps_in_cue.dim] * correlation), ir.dim, -1),
                dtype=dtype,
            )
            U_matrix_full = torch.moveaxis(U_matrix_full, -2, 0)
            if ir.dim == 1:
                U_matrix_full = U_matrix_full[0]
            U.append(U_matrix_full)
        return U

    class O3_e3nn(cue.O3):
        def __mul__(
            rep1: "O3_e3nn",
            rep2: "O3_e3nn",
        ) -> Iterator["O3_e3nn"]:
            return [O3_e3nn(l=ir.l, p=ir.p) for ir in cue.O3.__mul__(rep1, rep2)]

        @classmethod
        def clebsch_gordan(
            cls,
            rep1: "O3_e3nn",
            rep2: "O3_e3nn",
            rep3: "O3_e3nn",
        ) -> np.ndarray:
            rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)
            if rep1.p * rep2.p == rep3.p:
                return o3.wigner_3j(rep1.l, rep2.l, rep3.l).numpy()[None] * np.sqrt(rep3.dim)
            return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

        def __lt__(rep1: "O3_e3nn", rep2: "O3_e3nn") -> bool:
            rep2 = rep1._from(rep2)
            return (rep1.l, rep1.p) < (rep2.l, rep2.p)

        @classmethod
        def iterator(cls) -> Iterator["O3_e3nn"]:
            for l in itertools.count(0):
                yield O3_e3nn(l=l, p=1 * (-1) ** l)
                yield O3_e3nn(l=l, p=-1 * (-1) ** l)

else:

    class O3_e3nn:
        pass
