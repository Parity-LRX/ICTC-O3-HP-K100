"""Shared radial basis utilities with MACE-style polynomial cutoff."""

from __future__ import annotations

import math

import torch
from e3nn.math import soft_one_hot_linspace as _e3nn_soft_one_hot_linspace


def mace_polynomial_cutoff(
    x: torch.Tensor,
    r_max: float | torch.Tensor,
    p: int = 50,
) -> torch.Tensor:
    """MACE polynomial envelope, C^(p-1) smooth at the cutoff.

    The polynomial is 1 at r=0 and has zero value and zero derivatives up to
    order p - 1 at r=r_max.  This mirrors MACE's PolynomialCutoff formula.
    """
    p_int = int(p)
    p_f = float(p_int)
    # Use Python scalars instead of host->device tensors (torch.as_tensor) so this
    # forward is CUDA-graph capturable. tensor/scalar ops cast the scalar to
    # x.dtype, so the result is numerically identical to the tensor version.
    if isinstance(r_max, torch.Tensor):
        r = x / r_max
        within = (x < r_max).to(dtype=x.dtype)
    else:
        r_max_f = float(r_max)
        r = x / r_max_f
        within = (x < r_max_f).to(dtype=x.dtype)
    envelope = (
        1.0
        - ((p_f + 1.0) * (p_f + 2.0) / 2.0) * torch.pow(r, p_int)
        + p_f * (p_f + 2.0) * torch.pow(r, p_int + 1)
        - (p_f * (p_f + 1.0) / 2.0) * torch.pow(r, p_int + 2)
    )
    return envelope * within


def mace_radial_embedding(
    edge_length: torch.Tensor,
    *,
    r_max: float,
    number_of_basis: int,
    function_type: str,
    polynomial_cutoff_p: int | None = None,
    sqrt_num_basis_norm: bool = True,
) -> torch.Tensor:
    """MACE radial basis, optionally multiplied by the MACE polynomial envelope.

    ``function_type="bessel"`` follows mace-torch's ``BesselBasis``:
    ``sqrt(2 / r_max) * sin(n*pi*r/r_max) / r``. ``function_type="gaussian"``
    follows mace-torch's evenly-spaced Gaussian centers. Other basis names fall
    back to e3nn's ``soft_one_hot_linspace`` for backward compatibility.

    polynomial_cutoff_p=None: raw radial basis.
    polynomial_cutoff_p=int: apply MACE polynomial envelope of that order.
    """
    kind = str(function_type).lower()
    r_max_f = float(r_max)
    n_basis = int(number_of_basis)
    if kind == "bessel":
        weights = (
            math.pi
            / r_max_f
            * torch.linspace(1.0, float(n_basis), steps=n_basis, dtype=edge_length.dtype, device=edge_length.device)
        )
        emb = math.sqrt(2.0 / r_max_f) * torch.sin(edge_length.unsqueeze(-1) * weights) / edge_length.clamp_min(1e-12).unsqueeze(-1)
    elif kind == "gaussian":
        centers = torch.linspace(0.0, r_max_f, steps=n_basis, dtype=edge_length.dtype, device=edge_length.device)
        coeff = -0.5 / (r_max_f / float(max(n_basis - 1, 1))) ** 2
        emb = torch.exp(coeff * torch.pow(edge_length.unsqueeze(-1) - centers, 2))
    else:
        emb = _e3nn_soft_one_hot_linspace(
            edge_length,
            0.0,
            r_max_f,
            n_basis,
            basis=str(function_type),
            cutoff=True,
        )
    if polynomial_cutoff_p is not None:
        envelope = mace_polynomial_cutoff(edge_length, r_max_f, int(polynomial_cutoff_p))
        emb = emb * envelope.unsqueeze(-1)
    if sqrt_num_basis_norm:
        # Historical FSCETP sqrt(num_basis) scale -- a constant the first radial linear absorbs
        # during training. Set False for byte-literal correspondence with MACE's
        # BesselBasis x PolynomialCutoff.
        emb = emb.mul(math.sqrt(int(number_of_basis)))
    return emb


def soft_one_hot_linspace_mace_cutoff(
    x: torch.Tensor,
    start: float,
    end: float,
    number: int,
    *,
    basis: str = "gaussian",
    cutoff: bool = True,
    polynomial_cutoff_p: int = 50,
) -> torch.Tensor:
    """e3nn-compatible radial basis with MACE polynomial cutoff when requested."""
    emb = _e3nn_soft_one_hot_linspace(
        x,
        float(start),
        float(end),
        int(number),
        basis=str(basis),
        cutoff=False,
    )
    if not cutoff:
        return emb
    envelope = mace_polynomial_cutoff(x, float(end), int(polynomial_cutoff_p))
    return emb * envelope.unsqueeze(-1)
