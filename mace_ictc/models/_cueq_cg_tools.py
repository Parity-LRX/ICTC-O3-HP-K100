"""Local cuEquivariance projection helpers for MACE symmetric contractions.

This is a small local equivalent of ``mace.tools.cg_cueq_tools``. It lets
MACE-ICTC convert MACE/e3nn reduced-CG symmetric-contraction weights into
cuEquivariance's descriptor weight space without depending on a recent
``mace-torch`` installation.

SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from functools import cache
from typing import Optional

import numpy as np
from e3nn import o3

from mace_ictc.models._mace_cg import U_matrix_real

try:
    import cuequivariance as cue
    from cuequivariance.etc.linalg import round_to_sqrt_rational, triu_array

    CUEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    cue = None
    round_to_sqrt_rational = None
    triu_array = None
    CUEQ_AVAILABLE = False


def _require_cueq() -> None:
    if not CUEQ_AVAILABLE:
        raise RuntimeError(
            "reduced-CG cuEquivariance product projection requires cuequivariance"
        )


def symmetric_contraction_proj(
    irreps_in: "cue.Irreps",
    irreps_out: "cue.Irreps",
    degrees: tuple[int, ...],
) -> tuple["cue.EquivariantPolynomial", np.ndarray]:
    """Project original-MACE path weights into cuEq descriptor path weights."""
    _require_cueq()
    return symmetric_contraction_cached(irreps_in, irreps_out, tuple(degrees))


@cache
def symmetric_contraction_cached(
    irreps_in: "cue.Irreps",
    irreps_out: "cue.Irreps",
    degrees: tuple[int, ...],
) -> tuple["cue.EquivariantPolynomial", np.ndarray]:
    _require_cueq()
    assert min(degrees) > 0

    # poly1 matches MACE's recursive U-matrix symmetric contraction; poly2 is
    # cuEquivariance's reduced descriptor contraction. The projection maps
    # weights from poly1 order/space into poly2.
    poly1 = cue.EquivariantPolynomial.stack(
        [
            cue.EquivariantPolynomial.stack(
                [
                    _symmetric_contraction(irreps_in, irreps_out[i : i + 1], deg)
                    for deg in reversed(degrees)
                ],
                [True, False, False],
            )
            for i in range(len(irreps_out))
        ],
        [True, False, True],
    )

    poly2 = cue.descriptors.symmetric_contraction(irreps_in, irreps_out, degrees)
    a1, a2 = [
        np.concatenate(
            [
                _flatten(
                    _stp_to_matrix(d.symmetrize_operands(range(1, d.num_operands - 1))),
                    1,
                    None,
                )
                for _, d in pol.polynomial.operations
            ],
            axis=1,
        )
        for pol in [poly1, poly2]
    ]

    nonzeros = np.nonzero(np.any(a1 != 0, axis=0) | np.any(a2 != 0, axis=0))[0]
    a1, a2 = a1[:, nonzeros], a2[:, nonzeros]
    projection = a1 @ np.linalg.pinv(a2)
    projection = round_to_sqrt_rational(projection)
    np.testing.assert_allclose(a1, projection @ a2, atol=1e-7)
    return poly2, projection


def _flatten(
    x: np.ndarray,
    axis_start: Optional[int] = None,
    axis_end: Optional[int] = None,
) -> np.ndarray:
    x = np.asarray(x)
    if axis_start is None:
        axis_start = 0
    if axis_end is None:
        axis_end = x.ndim
    assert 0 <= axis_start <= axis_end <= x.ndim
    return x.reshape(
        x.shape[:axis_start]
        + (np.prod(x.shape[axis_start:axis_end]),)
        + x.shape[axis_end:]
    )


def _stp_to_matrix(d: "cue.SegmentedTensorProduct") -> np.ndarray:
    matrix = np.zeros([operand.num_segments for operand in d.operands])
    for path in d.paths:
        matrix[path.indices] = path.coefficients
    return matrix


def _symmetric_contraction(
    irreps_in: "cue.Irreps",
    irreps_out: "cue.Irreps",
    degree: int,
) -> "cue.EquivariantPolynomial":
    _require_cueq()
    mul = irreps_in.muls[0]
    assert all(mul == m for m in irreps_in.muls)
    assert all(mul == m for m in irreps_out.muls)
    irreps_in = irreps_in.set_mul(1)
    irreps_out = irreps_out.set_mul(1)

    input_operands = range(1, degree + 1)
    output_operand = degree + 1
    abc = "abcdefgh"[:degree]
    descriptor = cue.SegmentedTensorProduct.from_subscripts(
        f"u_{'_'.join(f'{a}' for a in abc)}_i+{abc}ui"
    )

    for operand in input_operands:
        descriptor.add_segment(operand, (irreps_in.dim,))

    irreps_in_e3nn = o3.Irreps(str(irreps_in))
    irreps_out_e3nn = o3.Irreps(str(irreps_out))
    for _, ir in irreps_out:
        u_matrix = U_matrix_real(
            irreps_in_e3nn,
            irreps_out_e3nn,
            int(degree),
            use_cueq_cg=True,
        )[-1]
        if str(ir) in {"0e", "0o"}:
            u_matrix = u_matrix.unsqueeze(0)
        u_array = np.asarray(u_matrix)
        u_array = np.moveaxis(u_array, 0, -1)

        if u_array.shape[-2] == 0:
            descriptor.add_segment(output_operand, {"i": ir.dim})
        else:
            descriptor.add_path(None, *(0,) * degree, None, c=triu_array(u_array, degree))

    descriptor = descriptor.flatten_coefficient_modes()
    descriptor = descriptor.append_modes_to_all_operands("u", {"u": mul})

    assert descriptor.num_operands >= 3
    [w_operand, x_operand], y_operand = descriptor.operands[:2], descriptor.operands[-1]
    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(irreps_in.new_scalars(w_operand.size), cue.ir_mul),
            cue.IrrepsAndLayout(mul * irreps_in, cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(mul * irreps_out, cue.ir_mul)],
        cue.SegmentedPolynomial(
            [w_operand, x_operand],
            [y_operand],
            [(cue.Operation([0] + [1] * degree + [2]), descriptor)],
        ),
    )
