"""Learned pairwise C6 dispersion (van der Waals) long-range term.

Completes the long-range physics alongside the multipole electrostatics: a degree-l
ICTC carrier gives the equivariant multipoles for electrostatics, while dispersion
needs only per-atom *invariant* coefficients (C6 is a scalar). The term is
E_disp = -1/2 sum_{i!=j} s6 * C6_ij / (r_ij^6 + R0_ij^6)  (Becke-Johnson-style damping,
smooth as r->0 so the short-range network owns the contact region), with the
geometric-mean combination C6_ij = sqrt(C6_i * C6_j) and R0_ij = R0_i + R0_j.

r^-6 is absolutely convergent in 3D -> a real-space pairwise sum suffices (no Ewald).
The energy depends only on edge lengths and per-atom scalars, so it is exactly
rotation- and translation-invariant by construction.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from mace_ictc.models.long_range import (
    _build_assignment_offsets,
    _dipole_pme_kernel_mesh,
    _dipole_pme_self_blocks,
    _prepare_frac_for_boundary,
    apply_periodic_dipole_pme_field,
    build_periodic_dipole_pme_kernel,
    build_periodic_dipole_pme_kernel_batched,
)
from mace_ictc.models.ictd_irreps import ictd_l2_to_rank2
from mace_ictc.utils.graph_utils import pbc_image_nmax


def _size_leq_zero(value) -> bool:
    """Trace-friendly size emptiness check for eager and symbolic-shape make_fx."""
    try:
        from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

        return bool(guard_size_oblivious(value <= 0))
    except Exception:
        return bool(value <= 0)


def _cell_volume(cell: torch.Tensor) -> torch.Tensor:
    return torch.linalg.det(cell).abs().clamp_min(1e-6)


def _quintic_smoothstep(x: torch.Tensor) -> torch.Tensor:
    """C2 smoothstep: 0 at x<=0, 1 at x>=1, zero first/second derivative at both ends."""
    t = x.clamp(0.0, 1.0)
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def _resolve_dispersion_switch_width(
    cutoff: float,
    min_cutoff: float,
    requested_width: float,
) -> float:
    width = float(requested_width)
    if width <= 0.0:
        return 0.0
    if cutoff <= 0.0:
        return width if min_cutoff > 0.0 else 0.0
    span = float(cutoff) - max(float(min_cutoff), 0.0)
    if span <= 0.0:
        return 0.0
    return max(0.0, min(width, 0.5 * span))


def _dispersion_edge_window(
    edge_lengths: torch.Tensor,
    *,
    cutoff: float,
    min_cutoff: float,
    switch_width: float,
) -> torch.Tensor:
    """Smooth edge weight for long-range dispersion cutoffs.

    The outer window removes the C6/MBD cutoff jump; the inner window replaces
    the old hard ``min_cutoff`` edge deletion. Both are C2, so energy and forces
    go continuously to zero at the boundaries.
    """
    weight = torch.ones_like(edge_lengths)
    width = _resolve_dispersion_switch_width(cutoff, min_cutoff, switch_width)
    if cutoff > 0.0:
        if width > 0.0:
            weight = weight * _quintic_smoothstep((float(cutoff) - edge_lengths) / width)
        else:
            weight = weight * (edge_lengths <= float(cutoff)).to(dtype=edge_lengths.dtype)
    if min_cutoff > 0.0:
        if width > 0.0:
            weight = weight * _quintic_smoothstep((edge_lengths - float(min_cutoff)) / width)
        else:
            weight = weight * (edge_lengths >= float(min_cutoff)).to(dtype=edge_lengths.dtype)
    return weight


def _estimate_dispersion_max_neighbors(
    num_atoms: int,
    cell: torch.Tensor,
    cutoff: float,
    *,
    pbc: bool,
    safety: float = 4.0,
    min_neighbors: int = 64,
) -> int:
    """Conservative radius-search cap that stays independent of system size.

    torch_cluster's radius search allocates against max_num_neighbors. Using N as
    that cap is correct but memory scales like O(N^2). For roughly homogeneous
    periodic systems, the expected degree is density * 4/3*pi*r^3, so use a
    safety factor and retry only through an explicit user override if needed.
    """
    if num_atoms <= 1:
        return 1
    cutoff_f = float(cutoff)
    if pbc:
        volume = float(_cell_volume(cell).detach().to(torch.float64).cpu().item())
    else:
        volume = max(cutoff_f**3, float(num_atoms))
    density = max(float(num_atoms) / max(volume, 1e-6), 1e-9)
    shell = (4.0 / 3.0) * 3.141592653589793 * cutoff_f**3
    expected = density * shell
    cap = int(max(float(min_neighbors), safety * expected + 16.0))
    return max(1, min(int(num_atoms), cap))


def _bump_neighbor_cap(cap: int, num_atoms: int) -> int:
    return max(cap + 1, min(int(num_atoms), int(cap) * 2))


def _normalize_max_num_neighbors(max_num_neighbors: int | None) -> int | None:
    if max_num_neighbors is None:
        return None
    value = int(max_num_neighbors)
    if value < 0:
        raise ValueError("dispersion max_num_neighbors must be >= 0 or None")
    return None if value == 0 else value


def _dense_neighbor_work_estimate(batch: torch.Tensor, cell: torch.Tensor, cutoff: float, *, pbc: bool) -> int:
    """Estimate dense neighbor-list work as max_g N_g^2 * n_images_g."""
    if batch.numel() == 0:
        return 0
    counts = torch.bincount(batch.to(torch.long), minlength=int(cell.shape[0]))
    max_work = 0
    for g in range(int(cell.shape[0])):
        m = int(counts[g].item())
        if m <= 0:
            continue
        image_count = 1
        if pbc:
            nmax = pbc_image_nmax(cell[g], float(cutoff), pbc=True).detach().cpu()
            image_count = int(torch.prod(2 * nmax + 1).item())
        max_work = max(max_work, m * m * image_count)
    return int(max_work)


def _dispersion_neighbor_complexity_context(
    *,
    method: str,
    cutoff: float,
    pbc: bool,
    max_graph_atoms: int,
    bruteforce_threshold: int,
    dense_work: int,
    dense_work_limit: int,
) -> str:
    return (
        f"method={method}, cutoff={float(cutoff):g}, pbc={bool(pbc)}, "
        f"max_graph_atoms={int(max_graph_atoms)}, "
        f"bruteforce_threshold={int(bruteforce_threshold)}, "
        f"dense_work={int(dense_work)}, dense_work_limit={int(dense_work_limit)}"
    )


def dispersion_cutoff_is_single_image_exact(cell: torch.Tensor, cutoff: float | torch.Tensor, *, pbc=True) -> bool:
    """Whether a nearest-image runtime dispersion graph can represent this cutoff exactly.

    This mirrors the mff/torch edge-sparse MBD deployment guard: for every
    periodic axis, ``2 * cutoff`` must not exceed the cell face height.  If this
    is false, exact MBD cutoff edges can include multiple periodic images or
    self-image couplings that the current LAMMPS nearest-image graph cannot
    represent; checkpoints trained with the ``pme_fft`` reciprocal-only MBD backend
    can deploy through the matching C++ use_fft solver for those cases.  Do not
    switch an edge-sparse checkpoint to pme_fft only at deployment time.
    """
    cell_mat = cell.reshape(-1, 3, 3)
    cutoff_t = torch.as_tensor(cutoff, device=cell_mat.device, dtype=cell_mat.dtype)
    for c in cell_mat:
        if bool((pbc_image_nmax(c, 2.0 * cutoff_t, pbc=pbc) > 1).any().item()):
            return False
    return True


def _lexicographic_positive(values: torch.Tensor) -> torch.Tensor:
    """First nonzero component is positive."""
    x = values[:, 0]
    y = values[:, 1]
    z = values[:, 2]
    return (x > 0) | ((x == 0) & (y > 0)) | ((x == 0) & (y == 0) & (z > 0))


def _canonical_undirected_shift_mask(src: torch.Tensor, dst: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """One representative of (src, dst, shift) ~ (dst, src, -shift)."""
    return (src < dst) | ((src == dst) & _lexicographic_positive(shifts))


def _canonical_undirected_edge_mask(src: torch.Tensor, dst: torch.Tensor, edge_vec: torch.Tensor) -> torch.Tensor:
    """Canonical coupling mask when only Cartesian edge vectors are available."""
    return (src < dst) | ((src == dst) & _lexicographic_positive(edge_vec))


def _stable_lexsort(columns: list[torch.Tensor]) -> torch.Tensor:
    if not columns:
        raise ValueError("stable lexsort requires at least one column")
    n = int(columns[0].numel())
    device = columns[0].device
    order = torch.arange(n, device=device, dtype=torch.long)
    for col in reversed(columns):
        try:
            local = torch.argsort(col.index_select(0, order), stable=True)
        except TypeError:
            local = torch.argsort(col.index_select(0, order))
        order = order.index_select(0, local)
    return order


def _sort_dispersion_edges(
    src: torch.Tensor,
    dst: torch.Tensor,
    shifts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic edge order for training/export parity and graph dumps."""
    if src.numel() <= 1:
        return src, dst, shifts
    order = _stable_lexsort([dst, src, shifts[:, 0], shifts[:, 1], shifts[:, 2]])
    return (
        src.index_select(0, order).contiguous(),
        dst.index_select(0, order).contiguous(),
        shifts.index_select(0, order).contiguous(),
    )


def _unique_sorted_dispersion_edges(
    src: torch.Tensor,
    dst: torch.Tensor,
    shifts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if src.numel() <= 1:
        return src, dst, shifts
    same_prev = (
        (src[1:] == src[:-1])
        & (dst[1:] == dst[:-1])
        & (shifts[1:, 0] == shifts[:-1, 0])
        & (shifts[1:, 1] == shifts[:-1, 1])
        & (shifts[1:, 2] == shifts[:-1, 2])
    )
    keep = torch.cat([torch.ones(1, dtype=torch.bool, device=src.device), ~same_prev], dim=0)
    return src[keep].contiguous(), dst[keep].contiguous(), shifts[keep].contiguous()


@torch.no_grad()
def normalize_dispersion_edges(
    src: torch.Tensor,
    dst: torch.Tensor,
    shifts: torch.Tensor,
    *,
    canonical_undirected: bool = False,
    sort_edges: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize explicit dispersion edges to the training/deployment convention.

    Explicit dataset edges are accepted for both pairwise and MBD dispersion.  Pairwise
    C6 keeps directed edges because its energy partition expects both directions.  MBD
    modes canonicalize either orientation to one undirected representative, including
    the positive half of self-image edges, matching the mff/torch deployment list.
    """
    src = src.to(dtype=torch.long)
    dst = dst.to(dtype=torch.long)
    shifts = shifts.round().to(dtype=torch.long)
    if canonical_undirected and src.numel() > 0:
        forward = _canonical_undirected_shift_mask(src, dst, shifts)
        reverse = _canonical_undirected_shift_mask(dst, src, -shifts)
        keep = forward | reverse
        flip = (~forward) & reverse
        src_new = torch.where(flip, dst, src)
        dst_new = torch.where(flip, src, dst)
        shifts_new = torch.where(flip.view(-1, 1), -shifts, shifts)
        src = src_new[keep]
        dst = dst_new[keep]
        shifts = shifts_new[keep]
        src, dst, shifts = _sort_dispersion_edges(src, dst, shifts)
        src, dst, shifts = _unique_sorted_dispersion_edges(src, dst, shifts)
        return src.contiguous(), dst.contiguous(), shifts.contiguous()
    if sort_edges:
        src, dst, shifts = _sort_dispersion_edges(src, dst, shifts)
    return src.contiguous(), dst.contiguous(), shifts.contiguous()


@torch.no_grad()
def _dispersion_neighbor_list_bruteforce(pos, batch, cell, cutoff, *, pbc=True, canonical_undirected=False):
    """Periodic pair list within ``cutoff`` (per graph), repo convention
    edge_vec = pos[i] - pos[j] + shifts @ cell, returning (src=j, dst=i, shifts).

    Pure torch (no torch_cluster); O(N_g^2 * images) per graph -> intended for the
    small/medium validation systems (production gets its list from LAMMPS/ASE). Only
    indices+integer shifts are produced here; recompute lengths from ``pos`` for
    differentiable forces. ``cutoff`` should be <~ min lattice length for periodic cells.
    small validation systems and fallback cases where a nearest-image cell list
    would not be semantically equivalent.  When ``canonical_undirected`` is true,
    return one representative of ``(src, dst, shift) ~ (dst, src, -shift)``;
    self-image couplings keep only the lexicographically positive shift half.
    """
    dev, dt = pos.device, pos.dtype
    cutoff_t = torch.as_tensor(float(cutoff), device=dev, dtype=dt)
    src_all, dst_all, shift_all = [], [], []
    for g in range(cell.shape[0]):
        idx = (batch == g).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        p = pos.index_select(0, idx)  # [m, 3]
        c = cell[g]                   # [3, 3]
        if pbc:
            nmax = pbc_image_nmax(c, cutoff_t, pbc=True)
            axes = [torch.arange(-int(nmax[a]), int(nmax[a]) + 1, device=dev) for a in range(3)]
        else:
            axes = [torch.zeros(1, dtype=torch.long, device=dev) for _ in range(3)]
        shifts = torch.cartesian_prod(*axes).to(dt)  # [S, 3]
        shift_vecs = shifts @ c                       # [S, 3]
        disp = p[:, None, None, :] - p[None, :, None, :] + shift_vecs[None, None, :, :]  # [i, j, S, 3]
        dist = disp.norm(dim=-1)                      # [i, j, S]
        mask = (dist > 1e-8) & (dist <= cutoff_t)
        ii, jj, ss = mask.nonzero(as_tuple=True)
        if ii.numel() == 0:
            continue
        src = idx[jj]
        dst = idx[ii]
        if canonical_undirected:
            keep = _canonical_undirected_shift_mask(src, dst, shifts[ss])
            if not bool(keep.any()):
                continue
            src = src[keep]
            dst = dst[keep]
            ss = ss[keep]
        src_all.append(src)
        dst_all.append(dst)
        shift_all.append(shifts[ss].to(torch.long))
    if not src_all:
        z = torch.zeros(0, dtype=torch.long, device=dev)
        return z, z, torch.zeros(0, 3, dtype=torch.long, device=dev)
    return torch.cat(src_all), torch.cat(dst_all), torch.cat(shift_all)


@torch.no_grad()
def _dispersion_neighbor_list_cell(
    pos,
    batch,
    cell,
    cutoff,
    *,
    pbc=True,
    canonical_undirected=False,
    allow_bruteforce_fallback: bool = True,
):
    """Cell-list dispersion neighbor builder.

    The periodic path uses a nearest-image convention. It is therefore selected
    only when the face-height image bound says one image shell is enough;
    otherwise the brute-force multi-image fallback is required for exact parity
    with the historical Python helper.
    """
    dev, dt = pos.device, pos.dtype
    cutoff_f = float(cutoff)
    cutoff_t = torch.as_tensor(cutoff_f, device=dev, dtype=dt)
    src_all, dst_all, shift_all = [], [], []
    offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]

    for g in range(cell.shape[0]):
        idx = (batch == g).nonzero(as_tuple=True)[0]
        m = int(idx.numel())
        if m == 0 or (m <= 1 and not pbc):
            continue
        p = pos.index_select(0, idx)
        c = cell[g]
        if pbc:
            lengths = c.norm(dim=-1).clamp_min(1e-6)
            # For larger cutoffs, multiple images of the same atom pair can be
            # inside the cutoff.  A nearest-image cell list would silently drop
            # those extra images, so keep the exact brute-force path.  Use the
            # face-height image bound instead of lattice-vector lengths; skewed
            # triclinic cells can have long vectors but much shorter face spacing.
            if not dispersion_cutoff_is_single_image_exact(c, cutoff_t, pbc=True):
                if not allow_bruteforce_fallback:
                    raise ImportError(
                        "Python cell-list dispersion neighbor builder would need the exact "
                        "multi-image brute-force fallback for this cell/cutoff. Install "
                        "torch_cluster, provide explicit dispersion edges, or set "
                        "allow_large_bruteforce_fallback=True for small validation systems."
                    )
                src, dst, shifts = _dispersion_neighbor_list_bruteforce(
                    pos.index_select(0, idx),
                    torch.zeros(m, dtype=torch.long, device=dev),
                    c.reshape(1, 3, 3),
                    cutoff_f,
                    pbc=True,
                    canonical_undirected=canonical_undirected,
                )
                if src.numel() == 0:
                    continue
                src_all.append(idx[src])
                dst_all.append(idx[dst])
                shift_all.append(shifts)
                continue
            inv_c = torch.linalg.inv(c)
            frac = p @ inv_c
            frac = frac - torch.floor(frac)
            nbin = torch.floor(lengths / cutoff_t).to(torch.long).clamp_min(1)
            coords = torch.floor(frac * nbin.to(dt)).to(torch.long)
            coords = torch.minimum(coords, (nbin - 1).view(1, 3))
        else:
            p_min = p.min(dim=0).values
            span = (p.max(dim=0).values - p_min).clamp_min(cutoff_t)
            nbin = torch.floor(span / cutoff_t).to(torch.long).clamp_min(1) + 1
            coords = torch.floor((p - p_min) / cutoff_t).to(torch.long)
            coords = torch.maximum(torch.zeros_like(coords), torch.minimum(coords, (nbin - 1).view(1, 3)))
            frac = None

        lin = (coords[:, 0] * nbin[1] + coords[:, 1]) * nbin[2] + coords[:, 2]
        order = torch.argsort(lin)
        lin_sorted = lin.index_select(0, order)
        unique, counts = torch.unique_consecutive(lin_sorted, return_counts=True)
        starts = torch.cumsum(torch.cat([counts.new_zeros(1), counts[:-1]]), dim=0)
        bins = {}
        for u, s, count in zip(unique.tolist(), starts.tolist(), counts.tolist()):
            bins[int(u)] = order.narrow(0, int(s), int(count))

        nx, ny, nz = (int(nbin[0]), int(nbin[1]), int(nbin[2]))

        def _linear(cx: int, cy: int, cz: int) -> int:
            return (cx * ny + cy) * nz + cz

        for lin_a, local_dst in bins.items():
            ax = lin_a // (ny * nz)
            ay = (lin_a // nz) % ny
            az = lin_a % nz
            seen_neighbors: set[int] = set()
            for ox, oy, oz in offsets:
                bx, by, bz = ax + ox, ay + oy, az + oz
                if pbc:
                    bx %= nx
                    by %= ny
                    bz %= nz
                elif bx < 0 or bx >= nx or by < 0 or by >= ny or bz < 0 or bz >= nz:
                    continue
                lin_b = _linear(bx, by, bz)
                if lin_b in seen_neighbors or lin_b not in bins:
                    continue
                seen_neighbors.add(lin_b)
                local_src = bins[lin_b]
                dst = idx.index_select(0, local_dst).view(-1, 1).expand(-1, int(local_src.numel())).reshape(-1)
                src = idx.index_select(0, local_src).view(1, -1).expand(int(local_dst.numel()), -1).reshape(-1)
                keep = src != dst
                if not bool(keep.any()):
                    continue
                src = src[keep]
                dst = dst[keep]
                if pbc:
                    frac_dst = frac.index_select(0, local_dst).view(-1, 1, 3).expand(
                        -1, int(local_src.numel()), -1
                    ).reshape(-1, 3)
                    frac_src = frac.index_select(0, local_src).view(1, -1, 3).expand(
                        int(local_dst.numel()), -1, -1
                    ).reshape(-1, 3)
                    shifts = -torch.round(frac_dst - frac_src).to(torch.long)
                    shifts = shifts[keep]
                    shift_vec = shifts.to(dt) @ c
                else:
                    shifts = torch.zeros(src.numel(), 3, dtype=torch.long, device=dev)
                    shift_vec = torch.zeros(src.numel(), 3, dtype=dt, device=dev)
                if canonical_undirected:
                    canonical = _canonical_undirected_shift_mask(src, dst, shifts)
                    if not bool(canonical.any()):
                        continue
                    src = src[canonical]
                    dst = dst[canonical]
                    shifts = shifts[canonical]
                    shift_vec = shift_vec[canonical]
                dist = (pos.index_select(0, dst) - pos.index_select(0, src) + shift_vec).norm(dim=-1)
                within = (dist > 1e-8) & (dist <= cutoff_t)
                if not bool(within.any()):
                    continue
                src_all.append(src[within])
                dst_all.append(dst[within])
                shift_all.append(shifts[within])

    if not src_all:
        z = torch.zeros(0, dtype=torch.long, device=dev)
        return z, z, torch.zeros(0, 3, dtype=torch.long, device=dev)
    return torch.cat(src_all), torch.cat(dst_all), torch.cat(shift_all)


@torch.no_grad()
def _dispersion_neighbor_list_torch_cluster(
    pos,
    batch,
    cell,
    cutoff,
    *,
    pbc=True,
    canonical_undirected=False,
    max_num_neighbors: int | None = None,
):
    """Build dispersion edges through torch_cluster's spatial radius search."""
    dev = pos.device
    cutoff_f = float(cutoff)
    src_all, dst_all, shift_all = [], [], []
    try:
        from torch_cluster import radius_graph
        from mace_ictc.utils.graph_utils import radius_graph_pbc_gpu
    except Exception as exc:  # noqa: BLE001 - fallback selection happens in caller
        raise ImportError("torch_cluster radius search is unavailable") from exc

    for g in range(cell.shape[0]):
        idx = (batch == g).nonzero(as_tuple=True)[0]
        m = int(idx.numel())
        if m == 0 or (m <= 1 and not pbc):
            continue
        p = pos.index_select(0, idx)
        c = cell[g]
        max_neighbors = (
            int(max_num_neighbors)
            if max_num_neighbors is not None
            else _estimate_dispersion_max_neighbors(m, c, cutoff_f, pbc=bool(pbc))
        )
        max_neighbors = max(1, min(m, max_neighbors))
        if pbc:
            while True:
                src, dst, shifts, saturated = radius_graph_pbc_gpu(
                    p,
                    cutoff_f,
                    c,
                    max_num_neighbors=max_neighbors,
                    pbc=True,
                    return_saturation=True,
                )
                if not saturated or max_neighbors >= m:
                    break
                max_neighbors = _bump_neighbor_cap(max_neighbors, m)
            shifts = shifts.round().to(torch.long)
        else:
            while True:
                edge_index = radius_graph(p, r=cutoff_f, loop=False, max_num_neighbors=max_neighbors)
                src, dst = edge_index[0], edge_index[1]
                saturated = False
                if max_neighbors > 0 and dst.numel() > 0:
                    counts = torch.bincount(dst, minlength=m)
                    saturated = bool((counts >= max_neighbors).any().item())
                if not saturated or max_neighbors >= m:
                    break
                max_neighbors = _bump_neighbor_cap(max_neighbors, m)
            shifts = torch.zeros(src.numel(), 3, dtype=torch.long, device=dev)
        if src.numel() == 0:
            continue
        src_g = idx.index_select(0, src.to(torch.long))
        dst_g = idx.index_select(0, dst.to(torch.long))
        if canonical_undirected:
            keep = _canonical_undirected_shift_mask(src_g, dst_g, shifts)
            if not bool(keep.any()):
                continue
            src_g = src_g[keep]
            dst_g = dst_g[keep]
            shifts = shifts[keep]
        src_all.append(src_g)
        dst_all.append(dst_g)
        shift_all.append(shifts.to(torch.long))

    if not src_all:
        z = torch.zeros(0, dtype=torch.long, device=dev)
        return z, z, torch.zeros(0, 3, dtype=torch.long, device=dev)
    return torch.cat(src_all), torch.cat(dst_all), torch.cat(shift_all)


@torch.no_grad()
def dispersion_neighbor_list(
    pos,
    batch,
    cell,
    cutoff,
    pbc=True,
    *,
    canonical_undirected: bool = False,
    sort_edges: bool = True,
    method: str = "auto",
    bruteforce_threshold: int = 1024,
    max_num_neighbors: int | None = None,
    allow_large_bruteforce_fallback: bool = False,
    return_info: bool = False,
):
    """Build a dispersion neighbor list.

    Args:
        canonical_undirected: return one representative of
            ``(src, dst, shift) ~ (dst, src, -shift)``.  Use this for MBD/SLQ-MBD
            so training matches the mff/torch deployment convention.  Pairwise C6
            should keep the default directed list because its energy partition
            assumes both directions and applies a 0.5 factor.
        sort_edges: return edges in deterministic ``dst, src, shift`` order.
        method: ``"auto"`` uses the dense GPU builder for small graphs and
            torch_cluster's radius search for larger graphs. If torch_cluster is
            unavailable, ``"auto"`` first tries the exact single-image Python
            cell-list path and only uses the dense fallback when
            ``allow_large_bruteforce_fallback=True``. ``"cell"`` selects the
            experimental sorted Python cell-list path, and ``"bruteforce"``
            selects the historical dense builder. Production deployment should
            still provide explicit edges from LAMMPS/Kokkos instead of rebuilding
            them in Python every step.
        bruteforce_threshold: largest per-graph atom count that ``"auto"`` sends
            to the dense builder, provided the image-expanded dense work
            ``N_g^2 * n_images`` also stays within the corresponding nearest-
            image budget.  On the 4090 test box, dense is faster at 512/1024
            atoms for normal boxes while torch_cluster is the viable path for
            larger systems or many-image triclinic/slab cases where the dense
            image tensor becomes memory-heavy.
        max_num_neighbors: optional radius-search cap for torch_cluster.  When
            omitted, the cap is estimated from density and cutoff instead of N,
            keeping large-system memory near O(N * local_degree).
        allow_large_bruteforce_fallback: if false, ``"auto"``/``"cell"`` paths
            raise a clear error when they would need the exact O(N^2) dense
            fallback for large or multi-image cases.
        return_info: if true, append a small metadata dict describing the
            selected builder path and dense-work estimate. The default return
            stays ``(src, dst, shifts)`` for existing callers.
    """
    if method not in {"auto", "cell", "bruteforce"}:
        raise ValueError(f"Unsupported dispersion neighbor-list method: {method!r}")
    if int(bruteforce_threshold) < 0:
        raise ValueError("dispersion bruteforce_threshold must be >= 0")
    max_num_neighbors = _normalize_max_num_neighbors(max_num_neighbors)

    def _finish(result):
        return _sort_dispersion_edges(*result) if sort_edges else result

    def _with_info(result, *, selected_method: str, **extra):
        if not return_info:
            return result
        info = {
            "requested_method": str(method),
            "selected_method": str(selected_method),
            "cutoff": float(cutoff),
            "pbc": bool(pbc),
            "bruteforce_threshold": int(bruteforce_threshold),
            "max_num_neighbors": max_num_neighbors,
            "canonical_undirected": bool(canonical_undirected),
        }
        info.update(extra)
        return (*result, info)

    if method == "auto":
        if batch.numel() == 0:
            max_graph_atoms = 0
        elif cell.shape[0] == 1:
            max_graph_atoms = int(batch.numel())
        else:
            max_graph_atoms = int(torch.bincount(batch.to(torch.long), minlength=int(cell.shape[0])).max().item())
        dense_work = _dense_neighbor_work_estimate(batch, cell, cutoff, pbc=bool(pbc))
        dense_work_limit = int(bruteforce_threshold) * int(bruteforce_threshold) * 27
        context = _dispersion_neighbor_complexity_context(
            method=method,
            cutoff=float(cutoff),
            pbc=bool(pbc),
            max_graph_atoms=max_graph_atoms,
            bruteforce_threshold=int(bruteforce_threshold),
            dense_work=dense_work,
            dense_work_limit=dense_work_limit,
        )
        if max_graph_atoms <= int(bruteforce_threshold) and dense_work <= dense_work_limit:
            result = _finish(
                _dispersion_neighbor_list_bruteforce(
                    pos, batch, cell, cutoff, pbc=pbc, canonical_undirected=canonical_undirected
                )
            )
            return _with_info(
                result,
                selected_method="auto_bruteforce",
                max_graph_atoms=max_graph_atoms,
                dense_work=dense_work,
                dense_work_limit=dense_work_limit,
            )
        try:
            result = _finish(
                _dispersion_neighbor_list_torch_cluster(
                    pos,
                    batch,
                    cell,
                    cutoff,
                    pbc=pbc,
                    canonical_undirected=canonical_undirected,
                    max_num_neighbors=max_num_neighbors,
                )
            )
            return _with_info(
                result,
                selected_method="auto_torch_cluster",
                max_graph_atoms=max_graph_atoms,
                dense_work=dense_work,
                dense_work_limit=dense_work_limit,
            )
        except ImportError as exc:
            try:
                result = _finish(
                    _dispersion_neighbor_list_cell(
                        pos,
                        batch,
                        cell,
                        cutoff,
                        pbc=pbc,
                        canonical_undirected=canonical_undirected,
                        allow_bruteforce_fallback=allow_large_bruteforce_fallback,
                    )
                )
                return _with_info(
                    result,
                    selected_method="auto_cell",
                    max_graph_atoms=max_graph_atoms,
                    dense_work=dense_work,
                    dense_work_limit=dense_work_limit,
                )
            except ImportError as cell_exc:
                if not allow_large_bruteforce_fallback:
                    raise ImportError(
                        "torch_cluster is required for exact auto dispersion neighbor lists "
                        "above bruteforce_threshold when the Python cell-list path would need "
                        "a large exact multi-image brute-force fallback; install torch_cluster, "
                        "provide explicit dispersion edges, or set method='bruteforce' only "
                        f"for small validation systems. Complexity context: {context}."
                    ) from cell_exc
            if not allow_large_bruteforce_fallback:
                raise ImportError(
                    "torch_cluster is required for auto dispersion neighbor lists above "
                    f"bruteforce_threshold={bruteforce_threshold}; install torch_cluster, "
                    "provide explicit dispersion edges, or set method='bruteforce' for "
                    f"small validation systems. Complexity context: {context}."
                ) from exc
            result = _finish(
                _dispersion_neighbor_list_bruteforce(
                    pos, batch, cell, cutoff, pbc=pbc, canonical_undirected=canonical_undirected
                )
            )
            return _with_info(
                result,
                selected_method="auto_bruteforce_fallback",
                max_graph_atoms=max_graph_atoms,
                dense_work=dense_work,
                dense_work_limit=dense_work_limit,
            )
    if method == "bruteforce":
        result = _finish(
            _dispersion_neighbor_list_bruteforce(
                pos, batch, cell, cutoff, pbc=pbc, canonical_undirected=canonical_undirected
            )
        )
        return _with_info(result, selected_method="bruteforce")
    result = _finish(
        _dispersion_neighbor_list_cell(
            pos,
            batch,
            cell,
            cutoff,
            pbc=pbc,
            canonical_undirected=canonical_undirected,
            allow_bruteforce_fallback=allow_large_bruteforce_fallback,
        )
    )
    return _with_info(result, selected_method="cell")


class PairwiseDispersion(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32, r0_floor: float = 0.5):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.r0_floor = float(r0_floor)
        self.c6_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.r0_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.s6 = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        node_feats: torch.Tensor,   # [N, feature_dim] per-atom INVARIANT descriptor
        edge_src: torch.Tensor,     # [E] sender j
        edge_dst: torch.Tensor,     # [E] receiver i
        edge_lengths: torch.Tensor, # [E] |r_ij|
        edge_weight: torch.Tensor | None = None,  # [E] smooth cutoff/min_cutoff window
    ) -> torch.Tensor:
        c6 = F.softplus(self.c6_head(node_feats)).squeeze(-1)               # [N] >0
        r0 = F.softplus(self.r0_head(node_feats)).squeeze(-1) + self.r0_floor  # [N] >0 (Angstrom)
        c6_ij = torch.sqrt((c6[edge_src] * c6[edge_dst]).clamp_min(0.0))    # geometric-mean rule
        r0_ij = r0[edge_src] + r0[edge_dst]
        r6 = edge_lengths.clamp_min(1e-6).pow(6)
        e_edge = -self.s6 * c6_ij / (r6 + r0_ij.pow(6))                     # BJ-damped, attractive
        if edge_weight is not None:
            e_edge = e_edge * edge_weight.to(dtype=e_edge.dtype)
        # directed edge list double-counts each pair -> 0.5; partition onto the receiver atom.
        per_atom = node_feats.new_zeros(node_feats.shape[0])
        per_atom.index_add_(0, edge_dst, 0.5 * e_edge)
        return per_atom.unsqueeze(-1)  # [N, 1]


class ManyBodyDispersion(nn.Module):
    """Isotropic QHO many-body dispersion baseline.

    Each atom gets a learned static polarizability alpha_i and oscillator
    frequency omega_i from invariant node features. For each graph, build the
    finite-range coupled-oscillator matrix

        C_ii = omega_i^2 I_3
        C_ij = s_MBD omega_i omega_j sqrt(alpha_i alpha_j) f_damp(r_ij) T_ij

    where T_ij = 3 rr/r^5 - I/r^3. The per-graph MBD energy is the zero-point
    energy shift 0.5 sum_p sqrt(lambda_p) - 1.5 sum_i omega_i, partitioned
    uniformly over atoms. This is O(N^3) and intended as a correctness baseline
    before approximate/deployment kernels.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 32,
        alpha_floor: float = 1.0e-4,
        omega_floor: float = 1.0e-3,
        eig_floor: float = 1.0e-8,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.alpha_floor = float(alpha_floor)
        self.omega_floor = float(omega_floor)
        self.eig_floor = float(eig_floor)
        self.alpha_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.omega_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        # Small initial coupling keeps early random models positive definite.
        self.coupling_scale = nn.Parameter(torch.tensor(0.03))
        self.beta_raw = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        node_feats: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_vec: torch.Tensor,
        num_graphs: int | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_atoms = node_feats.shape[0]
        per_atom = node_feats.new_zeros(n_atoms)
        if n_atoms == 0:
            return per_atom.unsqueeze(-1)

        alpha = F.softplus(self.alpha_head(node_feats)).squeeze(-1) + self.alpha_floor
        omega = F.softplus(self.omega_head(node_feats)).squeeze(-1) + self.omega_floor
        beta = F.softplus(self.beta_raw) + 1.0e-6
        coupling_scale = self.coupling_scale
        eye3 = torch.eye(3, dtype=node_feats.dtype, device=node_feats.device)

        num_graphs = int(num_graphs) if num_graphs is not None else (int(batch.max().item()) + 1 if batch.numel() else 0)
        for g in range(num_graphs):
            if num_graphs == 1:
                idx = torch.arange(n_atoms, dtype=torch.long, device=node_feats.device)
                m = n_atoms
                local = None
                same_graph = None
            else:
                idx = (batch == g).nonzero(as_tuple=True)[0]
                m = idx.numel()
                if m <= 1:
                    continue
                local = torch.full((n_atoms,), -1, dtype=torch.long, device=node_feats.device)
                local[idx] = torch.arange(m, dtype=torch.long, device=node_feats.device)

                same_graph = (batch[edge_src] == g) & (batch[edge_dst] == g) & (edge_src != edge_dst)
            # Directed neighbor lists normally contain i<-j and j<-i. T(r)=T(-r), so keep
            # one canonical orientation to avoid double-strength coupling. For the single-graph
            # AOTI path, keep the edge tensor length fixed and zero non-canonical couplings instead
            # of boolean-filtering to a data-dependent length.
            if same_graph is None:
                es = edge_src
                ed = edge_dst
                ev = edge_vec
                coupling_weight = _canonical_undirected_edge_mask(edge_src, edge_dst, edge_vec).to(dtype=node_feats.dtype)
                if edge_weight is not None:
                    coupling_weight = coupling_weight * edge_weight.to(dtype=node_feats.dtype)
            else:
                same_graph = same_graph & _canonical_undirected_edge_mask(edge_src, edge_dst, edge_vec)
                es = edge_src[same_graph]
                ed = edge_dst[same_graph]
                ev = edge_vec[same_graph]
                coupling_weight = None if edge_weight is None else edge_weight[same_graph].to(dtype=node_feats.dtype)

            cmat = torch.diag_embed(omega[idx].repeat_interleave(3).pow(2))
            li = ed if local is None else local[ed]
            lj = es if local is None else local[es]
            r = ev.norm(dim=-1).clamp_min(1.0e-6)
            rhat = ev / r.unsqueeze(-1)
            tensor = (3.0 * rhat.unsqueeze(-1) * rhat.unsqueeze(-2) - eye3) / r.pow(3).view(-1, 1, 1)
            radius = alpha[es].pow(1.0 / 3.0) + alpha[ed].pow(1.0 / 3.0) + 1.0e-6
            damp = 1.0 - torch.exp(-((r / (beta * radius)).clamp_min(0.0)).pow(6))
            pref = coupling_scale * omega[es] * omega[ed] * torch.sqrt((alpha[es] * alpha[ed]).clamp_min(0.0)) * damp
            if coupling_weight is not None:
                pref = pref * coupling_weight
            blocks = pref.view(-1, 1, 1) * tensor
            rows = (3 * li).unsqueeze(1) + torch.arange(3, device=node_feats.device).view(1, 3)
            cols = (3 * lj).unsqueeze(1) + torch.arange(3, device=node_feats.device).view(1, 3)
            cmat.index_put_((rows.unsqueeze(2), cols.unsqueeze(1)), blocks, accumulate=True)
            cmat.index_put_((cols.unsqueeze(2), rows.unsqueeze(1)), blocks.transpose(-1, -2), accumulate=True)

            eigvals = torch.linalg.eigvalsh(cmat).clamp_min(self.eig_floor)
            e_graph = 0.5 * eigvals.sqrt().sum() - 1.5 * omega[idx].sum()
            per_atom[idx] = e_graph / m
        return per_atom.unsqueeze(-1)


class _SqrtFirstMomentQuad(torch.autograd.Function):
    r"""SLQ Gauss-quadrature estimate ``e_1^T sqrt_smooth(T) e_1`` with a degeneracy-safe backward.

    For each batch element the estimate is the (0,0) entry of a smoothly-floored matrix square
    root of the symmetric tridiagonal ``T`` ([B,k,k])::

        est = e_1^T f(T) e_1 = sum_m (U[0,m])^2 f(lambda_m),
        f(lambda) = sqrt(clamp_min(0.5 (lambda + sqrt(lambda^2 + eps^2)), eps)),  eps = eig_floor.

    The forward diagonalizes ``T`` and contracts exactly like the previous in-line code, so the
    result is bit-identical to ``(evecs[:,0,:].square() * smooth.clamp_min(eps).sqrt()).sum(-1)``.

    The motivation for the custom backward: ``torch.linalg.eigh``'s built-in gradient routes through
    the eigenvector sensitivity ``F_ij = 1/(lambda_i - lambda_j)``, which is +-inf when two Ritz
    values coincide -- a real occurrence in MBD training when a large learned coupling drives nearly
    degenerate Lanczos spectra, producing a sudden NaN gradient on a single geometry. The correct
    gradient of a symmetric matrix function (Daleckii-Krein) uses the *divided difference* of ``f``
    rather than ``1/(lambda_i - lambda_j)`` and is therefore finite at degeneracy: as
    ``lambda_i -> lambda_j`` the divided difference tends to ``f'(lambda_i)``.
    """

    @staticmethod
    def forward(ctx, tri: torch.Tensor, eig_floor: float, eig_ceil: float = 1.0e4) -> torch.Tensor:
        eps = float(eig_floor)
        ceil = float(eig_ceil)
        # Empty batch: keep the [B] shape and a valid grad path (eigh on a 0-batch is fine, but
        # short-circuit to avoid relying on backend behaviour for B=0).
        if tri.shape[0] == 0:
            ctx.eps = eps
            ctx.save_for_backward(
                tri.new_zeros(0, tri.shape[-1]),               # evals
                tri.new_zeros(0, tri.shape[-1], tri.shape[-1]),  # evecs
                tri.new_zeros(0, tri.shape[-1]),               # f
                tri.new_zeros(0, tri.shape[-1]),               # fprime
                tri.new_zeros(0, tri.shape[-1]),               # a
            )
            return tri.new_zeros(0)
        evals, evecs = torch.linalg.eigh(tri)                  # [B,k], [B,k,k]
        smooth = 0.5 * (evals + torch.sqrt(evals.square() + eps * eps))
        g = smooth.clamp_min(eps)
        # smooth spectral CEILING (mirrors the eps floor): gc = ceil*g/(ceil+g) == g for g<<ceil
        # (physical region ~O(omega^2), accuracy-neutral) and saturates -> ceil for g>>ceil. A
        # runaway learned coupling driving Ritz values huge then gets fprime->0 (no push) and a
        # finite sqrt(C), so grad-clip recovers instead of cascading (polarization-catastrophe guard).
        gc = (ceil * g) / (ceil + g)
        f = gc.sqrt()                       # [B,k] = f(lambda)
        a = evecs[:, 0, :]                                     # [B,k] first row U[0,:]
        # smooth'(lambda) = 0.5 (1 + lambda / sqrt(lambda^2 + eps^2)); g'(lambda) = smooth' where
        # smooth > eps else 0 (the clamp_min has zero slope on the floored branch).  f'(lambda) =
        # g'(lambda) / (2 f(lambda)); f >= sqrt(eps) > 0 so the division is safe.
        smooth_grad = 0.5 * (1.0 + evals / torch.sqrt(evals.square() + eps * eps))
        gprime = torch.where(smooth > eps, smooth_grad, torch.zeros_like(smooth_grad))
        gc_grad = (ceil * ceil) / ((ceil + g) * (ceil + g))    # dgc/dg = ceil^2/(ceil+g)^2 -> 0 for g>>ceil
        fprime = (gc_grad * gprime) / (2.0 * f)                            # [B,k] = f'(lambda)
        ctx.eps = eps
        ctx.save_for_backward(evals, evecs, f, fprime, a)
        return (a.square() * f).sum(dim=-1)                    # [B]

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # grad_out: [B]
        evals, evecs, f, fprime, a = ctx.saved_tensors
        if evals.shape[0] == 0:
            return torch.zeros_like(evecs).new_zeros(evecs.shape), None, None

        k = evals.shape[-1]
        # Divided differences  f1_ij = (f_i - f_j) / (lambda_i - lambda_j) for i != j, and
        # f1_ii = f'(lambda_i).  Build it degeneracy-safe with the where/safe-denominator trick so
        # no 0/0 enters the autograd graph (this Function is itself differentiable, e.g. for
        # gradcheck and double-backward through the divided difference).
        lam_i = evals.unsqueeze(-1)                            # [B,k,1]
        lam_j = evals.unsqueeze(-2)                            # [B,1,k]
        f_i = f.unsqueeze(-1)
        f_j = f.unsqueeze(-2)
        denom = lam_i - lam_j                                  # [B,k,k]
        # tol in the eigenvalue scale: 1e-7 absolute is appropriate for the O(1)-O(omega^2) Ritz
        # values seen here; scale-free enough for the fp64 tests and the fp32 training spectrum.
        tol = 1.0e-7
        near = denom.abs() <= tol
        denom_safe = torch.where(near, torch.ones_like(denom), denom)
        diff = (f_i - f_j) / denom_safe
        fprime_diag = fprime.unsqueeze(-1).expand(-1, k, k)    # broadcast f'(lambda_i) along j
        f1 = torch.where(near, fprime_diag, diff)              # [B,k,k]

        # M_ij = a_i a_j f1_ij ; G = U M U^T ; dest/dT = 0.5 (G + G^T), scaled by grad_out.
        M = a.unsqueeze(-1) * a.unsqueeze(-2) * f1             # [B,k,k]
        G = torch.matmul(torch.matmul(evecs, M), evecs.transpose(-1, -2))  # [B,k,k]
        grad_tri = 0.5 * (G + G.transpose(-1, -2)) * grad_out.view(-1, 1, 1)
        return grad_tri, None, None


class ManyBodyDispersionSLQ(nn.Module):
    """Matrix-free stochastic-Lanczos QHO many-body dispersion.

    This approximates the dense MBD zero-point energy

        0.5 Tr sqrt(C) - 1.5 sum_i omega_i

    without constructing ``C`` or diagonalizing the full ``3N x 3N`` matrix. The
    expensive operation is a matrix-vector product, available in two operator
    backends (select via ``operator_backend``):

      * ``edge_sparse`` (default): assembles the product from the cutoff dispersion
        edge list -- cost O(num_probes * lanczos_steps * E_disp); cutoff-truncated,
        O(E) and fast for small/medium systems.
      * ``pme_fft``: a reciprocal-only PME matvec (spread -> FFT -> screened dipole
        kernel -> iFFT -> gather) that bypasses the real-space dispersion graph.

    Both backends deploy: the LAMMPS/AOTI C++ MBD solver mirrors each operator
    exactly (edge_sparse = direct damp*T_bare edge sum; pme_fft = the use_fft
    reciprocal PME matching apply_periodic_dipole_pme_field to ~1e-10), so a model
    trained with either backend deploys consistently -- MATCH the backend across
    train and deploy.  The deterministic Rademacher probes keep training and force
    labels reproducible.
    """

    AVAILABLE_OPERATOR_BACKENDS = {"edge_sparse", "pme_fft"}
    RESERVED_OPERATOR_BACKENDS: set[str] = set()

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 32,
        alpha_floor: float = 1.0e-4,
        omega_floor: float = 1.0e-3,
        eig_floor: float = 1.0e-6,
        eig_ceil: float = 1.0e30,
        pd_rescale: bool = True,
        pd_margin: float = 0.1,
        pd_power_iters: int = 10,
        pd_amortize: bool = True,
        mbd_static_batch: bool = False,
        trace_estimator: str = "hutchinson",
        hutchpp_sketch: int = 4,
        hutchpp_residual: int = 2,
        num_probes: int = 8,
        lanczos_steps: int = 16,
        probe_mode: str = "rademacher",
        quadrature: str = "eigh",
        sqrt_iterations: int = 8,
        operator_backend: str = "edge_sparse",
        pme_mesh_size: int = 32,
        pme_assignment: str = "pcs",
        pme_k_norm_floor: float = 1.0e-6,
        pme_assignment_window_floor: float = 1.0e-6,
        pme_ewald_alpha_prefactor: float = 5.0,
        anisotropic_polarizability: bool = False,
        learnable_energy_scale: bool = True,
    ) -> None:
        super().__init__()
        if probe_mode not in {"rademacher", "atom-rademacher", "basis"}:
            raise ValueError(f"Unsupported SLQ probe mode: {probe_mode!r}")
        if quadrature not in {"eigh", "newton-schulz"}:
            raise ValueError(f"Unsupported SLQ quadrature: {quadrature!r}")
        if operator_backend in self.RESERVED_OPERATOR_BACKENDS:
            raise NotImplementedError(
                "SLQ-MBD operator_backend='pme_fft' is reserved for the reciprocal-space "
                "PME/cuFFT dipole-tensor matvec; use 'edge_sparse' until that backend is implemented."
            )
        if operator_backend not in self.AVAILABLE_OPERATOR_BACKENDS:
            raise ValueError(
                f"Unsupported SLQ-MBD operator backend {operator_backend!r}; "
                f"available backends: {sorted(self.AVAILABLE_OPERATOR_BACKENDS)}; "
                f"reserved backends: {sorted(self.RESERVED_OPERATOR_BACKENDS)}"
            )
        self.feature_dim = int(feature_dim)
        self.alpha_floor = float(alpha_floor)
        self.omega_floor = float(omega_floor)
        self.eig_floor = float(eig_floor)
        self.eig_ceil = float(eig_ceil)
        self.pd_rescale = bool(pd_rescale)
        self.pd_margin = float(pd_margin)
        self.pd_power_iters = int(pd_power_iters)
        self.pd_amortize = bool(pd_amortize)
        self.mbd_static_batch = bool(mbd_static_batch)
        self.trace_estimator = str(trace_estimator)
        self.hutchpp_sketch = int(hutchpp_sketch)
        self.hutchpp_residual = int(hutchpp_residual)
        self.num_probes = int(num_probes)
        self.lanczos_steps = int(lanczos_steps)
        self.probe_mode = str(probe_mode)
        self.quadrature = str(quadrature)
        self.sqrt_iterations = int(sqrt_iterations)
        self.operator_backend = str(operator_backend)
        self.pme_mesh_size = int(pme_mesh_size)
        self.pme_assignment = str(pme_assignment)
        self.pme_k_norm_floor = float(pme_k_norm_floor)
        self.pme_assignment_window_floor = float(pme_assignment_window_floor)
        self.pme_ewald_alpha_prefactor = float(pme_ewald_alpha_prefactor)
        self.anisotropic_polarizability = bool(anisotropic_polarizability)
        self.register_buffer(
            "pme_assignment_offsets",
            _build_assignment_offsets(self.pme_assignment),
            persistent=False,
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.omega_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        # AOTInductor-safe constants as buffers: a buffer moves with module.to(device) and the export
        # treats it as an on-device input, whereas torch.eye(...,device=) / _harmonic_basis (built on CPU
        # then .to'd) get const-folded onto CPU -> the GPU Triton kernels can't read them ("cpu tensor"
        # autotuning failure). Non-persistent: not in the state_dict (no checkpoint-compat change).
        self.register_buffer("_eye3", torch.eye(3, dtype=torch.float64), persistent=False)
        if self.anisotropic_polarizability:
            # equivariant l=2 readout: channel-mix the l=2 node block -> one l=2 tensor (5D), then
            # ictd_l2_to_rank2 -> 3x3 traceless symmetric anisotropy; an l=0 gate bounds it for PD.
            self.l2_mix = nn.Linear(self.feature_dim, 1, bias=False)
            self.l2_gate = nn.Linear(self.feature_dim, 1)
            # 5 basis matrices: ictd_l2_to_rank2(c) == einsum("nk,kij->nij", c, l2_basis) by linearity.
            self.register_buffer(
                "_l2_basis", ictd_l2_to_rank2(torch.eye(5, dtype=torch.float64)), persistent=False)
        self.coupling_scale = nn.Parameter(torch.tensor(0.03))
        self.beta_raw = nn.Parameter(torch.tensor(1.0))
        # Gentle-start via a SCHEDULED RAMP, not a learnable init-0 gate. The learnable init-0 gate
        # (mirroring electrostatics) SELF-LOCKED for MBD: with the contribution gated to ~0 the
        # polarizability heads got ~0 gradient, so energy_scale never grew -- the carbon run ended
        # at energy_scale=0.0045 => MBD effectively OFF (a confounded null result). Instead: init
        # energy_scale=1.0 and force a linear 0->1 warmup over the first `_mbd_ramp_steps` TRAINING
        # forwards. This keeps the dangerous first ~10 steps small (ramp=0.033 at step 10 vs the
        # un-gated 1.0 that diverged to NaN in B_isoMBD) BUT guarantees the dispersion turns fully
        # on so the heads actually train. After the ramp, energy_scale is free to learn (stays ~1
        # => MBD used; ->0 => genuinely not needed -- an UN-confounded signal). At eval the ramp is
        # bypassed (full strength). A loaded checkpoint overrides the energy_scale init.
        # The ramp counter is persistent: resume must not silently re-disable MBD for another
        # ramp window after the checkpoint already trained past it.
        # learnable_energy_scale=False PINS energy_scale at a fixed 1.0 (registered as a buffer, not
        # a Parameter -> absent from model.parameters(), the optimizer cannot move it). Diagnostic for
        # the carbon null result (2026-07-01): with it learnable, gradient descent settled it at 0.056
        # (heads starved) even after the ramp guaranteed full gradient exposure -- pinning it forces the
        # alpha/omega/l2 heads to adapt AROUND a mandatory full-strength MBD term instead of being able
        # to suppress it. Either way the key name "energy_scale" is unchanged in the state_dict (a buffer
        # is included in state_dict same as a Parameter), so old/new checkpoints load into either mode.
        self.learnable_energy_scale = bool(learnable_energy_scale)
        if self.learnable_energy_scale:
            self.energy_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("energy_scale", torch.tensor(1.0))
        self._mbd_ramp_steps = 300
        self._mbd_ramp_count_value = 0
        self.register_buffer("_mbd_ramp_count_state", torch.zeros((), dtype=torch.long), persistent=True)

    @property
    def _mbd_ramp_count(self) -> int:
        return self._mbd_ramp_count_value

    @_mbd_ramp_count.setter
    def _mbd_ramp_count(self, value: int) -> None:
        self._mbd_ramp_count_value = int(value)
        self._mbd_ramp_count_state.fill_(self._mbd_ramp_count_value)

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        self._mbd_ramp_count_value = int(self._mbd_ramp_count_state.detach().cpu().item())

    def complete_training_ramp(self) -> None:
        """Force the training ramp to full strength before tracing/static export paths."""
        self._mbd_ramp_count = self._mbd_ramp_steps

    def polarizability_factor(self, node_feats: torch.Tensor, l2_feats: torch.Tensor | None) -> torch.Tensor:
        """Per-atom symmetric positive-definite 3x3 factor B = alpha^{1/2} from the ICTC features.
        Isotropic: B = sqrt(alpha) * I (l=0 only). Anisotropic: B = b0 * (I + D), D = gate * (l=2 -> 3x3
        traceless) bounded so ||D|| < gate < 1 (PD guaranteed). Equivariant: B(R) = R B R^T."""
        # b0 = sqrt(alpha_iso): the isotropic code uses wsa=omega*sqrt(alpha), so the factor B=sqrt(alpha)*I.
        b0 = (F.softplus(self.alpha_head(node_feats)).squeeze(-1) + self.alpha_floor).sqrt()  # [N]
        eye = self._eye3.to(node_feats.dtype)                                          # [3,3] on-device buffer
        if not self.anisotropic_polarizability or l2_feats is None:
            return b0.view(-1, 1, 1) * eye                                            # [N,3,3] isotropic
        t = self.l2_mix(l2_feats.transpose(1, 2)).squeeze(-1)                          # [N,5] one l=2 tensor
        d_raw = torch.einsum("nk,kij->nij", t, self._l2_basis.to(t.dtype))             # [N,3,3] traceless sym, equivariant
        gate = 0.9 * torch.sigmoid(self.l2_gate(node_feats)).squeeze(-1)              # [N] in (0, 0.9)
        # SMOOTH Frobenius norm sqrt(sum_sq + eps). torch.norm(...).clamp_min has backward x/||x|| (1st
        # order) and ~1/||x||^3 (2nd order); clamp_min floors the VALUE but not those backward divisions,
        # so when d_raw -> 0 (the near-zero-init l2_mix at from-scratch step 0) the force-loss param
        # gradient (2nd order through this op) blows to NaN and corrupts every weight at optimizer.step().
        # sqrt(sum_sq + eps) is value-identical for ||d_raw|| >> 1e-6 yet its 1st/2nd-order backward stay
        # bounded by ~1/sqrt(eps); the floor only touches the unphysical ||d_raw|| -> 0 region (D -> 0 there
        # anyway). This is the anisotropic-MBD stability fix: forward AND 2nd-order finite for any d_raw.
        dn = (d_raw.flatten(1).pow(2).sum(dim=1) + 1.0e-12).sqrt()                      # [N] smooth ||d_raw||_F
        D = (gate / (1.0 + dn)).view(-1, 1, 1) * d_raw                                 # ||D||_F < gate < 1
        return b0.view(-1, 1, 1) * (eye + D)                                           # [N,3,3] sym PD

    def emit_source(self, node_feats: torch.Tensor, l2_feats: torch.Tensor | None = None) -> torch.Tensor:
        """Deploy path: per-atom MBD source for the C++ solver; the coupled-dipole energy is DEFERRED to
        C++ (no double count). Isotropic: [omega, alpha] [N,2]. Anisotropic:
        [omega, alpha_iso, Bxx, Byy, Bzz, Bxy, Bxz, Byz] [N,8]."""
        omega = F.softplus(self.omega_head(node_feats)).squeeze(-1) + self.omega_floor
        alpha = F.softplus(self.alpha_head(node_feats)).squeeze(-1) + self.alpha_floor    # alpha_iso (=b0^2)
        if not self.anisotropic_polarizability:
            return torch.stack([omega, alpha], dim=-1)                                    # [N,2]
        b = self.polarizability_factor(node_feats, l2_feats)                              # [N,3,3] sym B=alpha^{1/2}
        # [omega, alpha_iso (for the damping radius), 6 unique B components] -- the C++ rebuilds W=omega*B.
        return torch.stack([omega, alpha, b[:, 0, 0], b[:, 1, 1], b[:, 2, 2],
                            b[:, 0, 1], b[:, 0, 2], b[:, 1, 2]], dim=-1)                  # [N,8]

    def mbd_beta(self) -> float:
        return float(F.softplus(self.beta_raw) + 1.0e-6)

    def mbd_coupling_scale(self) -> float:
        return float(self.coupling_scale.detach())

    def mbd_energy_scale(self) -> float:
        return float(self.energy_scale.detach().cpu().item())

    def _build_edge_sparse_matvec(
        self,
        *,
        omega_local: torch.Tensor,
        li: torch.Tensor,
        lj: torch.Tensor,
        blocks: torch.Tensor,
    ):
        def matvec(v: torch.Tensor) -> torch.Tensor:
            y = omega_local.square().view(1, -1, 1) * v
            v_j = v.index_select(1, lj)
            v_i = v.index_select(1, li)
            contrib_i = torch.matmul(blocks.unsqueeze(0), v_j.unsqueeze(-1)).squeeze(-1)
            contrib_j = torch.matmul(blocks.transpose(-1, -2).unsqueeze(0), v_i.unsqueeze(-1)).squeeze(-1)
            idx_i = li.view(1, -1, 1).expand(v.size(0), -1, 3)
            idx_j = lj.view(1, -1, 1).expand(v.size(0), -1, 3)
            y = y.scatter_add(1, idx_i, contrib_i)
            y = y.scatter_add(1, idx_j, contrib_j)
            return y

        return matvec

    def _build_pme_fft_matvec(
        self,
        *,
        pos_local: torch.Tensor,
        cell: torch.Tensor,
        alpha_local: torch.Tensor,
        omega_local: torch.Tensor,
        coupling_scale: torch.Tensor,
        pme_kernel: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        w_local: torch.Tensor | None = None,
    ):
        if self.pme_mesh_size <= 0:
            raise ValueError("pme_mesh_size must be positive")
        frac = _prepare_frac_for_boundary(
            pos_local,
            cell,
            boundary="periodic",
            slab_padding_factor=1,
        )
        mesh_size = int(self.pme_mesh_size)
        if w_local is None:  # isotropic fallback: W = omega*sqrt(alpha)*I
            local_scale = omega_local * alpha_local.clamp_min(0.0).sqrt()
            w_local = local_scale.view(-1, 1, 1) * torch.eye(3, dtype=pos_local.dtype, device=pos_local.device)
        if pme_kernel is None:
            k_cart, k2, spectral = build_periodic_dipole_pme_kernel(
                cell=cell,
                mesh_size=mesh_size,
                assignment=self.pme_assignment,
                device=pos_local.device,
                dtype=pos_local.dtype,
                k_norm_floor=self.pme_k_norm_floor,
                assignment_window_floor=self.pme_assignment_window_floor,
                ewald_alpha_prefactor=self.pme_ewald_alpha_prefactor,
            )
        else:
            k_cart, k2, spectral = pme_kernel
        kernel_mesh_flat = _dipole_pme_kernel_mesh(
            mesh_size=mesh_size,
            k_cart=k_cart,
            k2=k2,
            spectral=spectral,
            k_norm_floor=self.pme_k_norm_floor,
        )
        self_blocks = _dipole_pme_self_blocks(
            frac,
            mesh_size=mesh_size,
            assignment=self.pme_assignment,
            assignment_offsets=self.pme_assignment_offsets,
            kernel_mesh_flat=kernel_mesh_flat,
        )

        def matvec(v: torch.Tensor) -> torch.Tensor:
            y = omega_local.square().view(1, -1, 1) * v
            # dipoles_i = W_i v_i  (3x3 factor; isotropic W=omega*sqrt(alpha)*I reproduces the scalar form)
            dipoles = torch.einsum("mab,pmb->pma", w_local, v).permute(1, 0, 2)
            field = apply_periodic_dipole_pme_field(
                frac,
                dipoles,
                mesh_size=mesh_size,
                assignment=self.pme_assignment,
                assignment_offsets=self.pme_assignment_offsets,
                k_cart=k_cart,
                k2=k2,
                spectral=spectral,
                k_norm_floor=self.pme_k_norm_floor,
                kernel_mesh_flat=kernel_mesh_flat,
                self_blocks=self_blocks,
            )
            field = field.permute(1, 0, 2)
            return y + coupling_scale * torch.einsum("mab,pmb->pma", w_local, field)

        return matvec

    def _build_matvec(
        self,
        *,
        omega_local: torch.Tensor,
        alpha_local: torch.Tensor,
        li: torch.Tensor,
        lj: torch.Tensor,
        blocks: torch.Tensor,
        pos_local: torch.Tensor | None,
        cell: torch.Tensor | None,
        coupling_scale: torch.Tensor,
        pme_kernel: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        w_local: torch.Tensor | None = None,
    ):
        if self.operator_backend == "edge_sparse":
            return self._build_edge_sparse_matvec(
                omega_local=omega_local,
                li=li,
                lj=lj,
                blocks=blocks,
            )
        if self.operator_backend == "pme_fft":
            if pos_local is None or cell is None:
                raise ValueError("SLQ-MBD operator_backend='pme_fft' requires pos and cell")
            return self._build_pme_fft_matvec(
                pos_local=pos_local,
                cell=cell,
                alpha_local=alpha_local,
                omega_local=omega_local,
                coupling_scale=coupling_scale,
                pme_kernel=pme_kernel,
                w_local=w_local,
            )
        raise RuntimeError(f"Unhandled SLQ-MBD operator backend {self.operator_backend!r}")

    def _make_probes(self, m: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        dim = 3 * m
        if _size_leq_zero(dim):
            return torch.zeros(0, 0, 3, device=device, dtype=dtype)
        if self.probe_mode == "basis":
            return torch.eye(dim, device=device, dtype=dtype).reshape(dim, m, 3)
        n_probe = max(int(self.num_probes), 1)
        if self.probe_mode == "atom-rademacher":
            probe_idx = torch.arange(n_probe, device=device, dtype=torch.long).view(n_probe, 1)
            atom_idx = torch.arange(m, device=device, dtype=torch.long).view(1, m)
            h = (
                (probe_idx.to(dtype=dtype) + 1.0) * 12.9898
                + (atom_idx.to(dtype=dtype) + 1.0) * 78.233
                + (probe_idx.to(dtype=dtype) + 1.0) * (atom_idx.to(dtype=dtype) + 1.0) * 0.137
            )
            signs = torch.where(torch.sin(h) >= 0.0, 1.0, -1.0).to(dtype=dtype)
            eye = torch.eye(3, device=device, dtype=dtype)
            return (signs[:, None, :, None] * eye[None, :, None, :]).reshape(3 * n_probe, m, 3)
        probe_idx = torch.arange(n_probe, device=device, dtype=torch.long).view(n_probe, 1, 1)
        atom_idx = torch.arange(m, device=device, dtype=torch.long).view(1, m, 1)
        comp_idx = torch.arange(3, device=device, dtype=torch.long).view(1, 1, 3)
        # Deterministic sinusoidal hash -> Rademacher signs. Avoid integer-parity hashes here:
        # they can collapse to only a few unique probe rows for regular atom/component grids.
        h = (
            (probe_idx.to(dtype=dtype) + 1.0) * 12.9898
            + (atom_idx.to(dtype=dtype) + 1.0) * 78.233
            + (comp_idx.to(dtype=dtype) + 1.0) * 37.719
            + (probe_idx.to(dtype=dtype) + 1.0) * (atom_idx.to(dtype=dtype) + 1.0) * 0.137
            + (probe_idx.to(dtype=dtype) + 1.0) * (comp_idx.to(dtype=dtype) + 1.0) * 0.193
        )
        signs = torch.where(torch.sin(h) >= 0.0, 1.0, -1.0)
        return signs.to(dtype=dtype)

    def _pd_rho(self, matvec, omega_diag: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
        """Estimate rho(M) = max|eigenvalue| of the NORMALIZED coupling
        ``M = Omega^-1 (C - Omega^2) Omega^-1`` by power iteration on the matvec (no_grad).

        C = Omega^2 + coupling is strictly positive-definite iff
        ``lambda_min(Omega^-1 C Omega^-1) = 1 + lambda_min(M) > 0`` i.e. ``rho(M) < 1``.
        The caller uses the returned rho to scale the coupling so rho stays < 1 - pd_margin
        => C stays PD => sqrt(C) gradient stays bounded => no polarization catastrophe.
        This is the physical MBD self-consistent screening, applied only when the learned
        coupling would otherwise drive C indefinite; in the healthy regime rho < 1 and the
        caller leaves the operator untouched (zero bias).

        ``omega_diag`` is Omega broadcast over ``v0`` (entries <= 0 are padding atoms and get a
        zero inverse, so they contribute nothing). The per-row norm reduces the last two dims
        (atoms, components); leading dims (probe / graph) are independent batch rows.
        """
        inv_omega = torch.where(omega_diag > 0, omega_diag.reciprocal(), torch.zeros_like(omega_diag))

        def Mv(v: torch.Tensor) -> torch.Tensor:
            cu = torch.nan_to_num(matvec(v * inv_omega), nan=0.0, posinf=1.0e30, neginf=-1.0e30)
            return cu * inv_omega - v

        with torch.no_grad():
            wn = v0.norm(dim=(-2, -1), keepdim=True).clamp_min(1.0e-14)
            v = v0 / wn
            for _ in range(max(int(self.pd_power_iters), 1)):
                w = Mv(v)
                wn = w.norm(dim=(-2, -1), keepdim=True).clamp_min(1.0e-14)
                v = w / wn
        return wn.squeeze(-1).squeeze(-1)  # ||M v|| for unit v -> rho(M), per probe row

    def _sqrt_first_moment(self, tri: torch.Tensor) -> torch.Tensor:
        if self.quadrature == "newton-schulz":
            eye = torch.eye(tri.size(-1), dtype=tri.dtype, device=tri.device).unsqueeze(0).expand_as(tri)
            scale = tri.norm(dim=(-2, -1)).clamp_min(self.eig_floor)
            y = tri / scale.view(-1, 1, 1)
            z = eye
            for _ in range(max(int(self.sqrt_iterations), 1)):
                t = 0.5 * (3.0 * eye - torch.matmul(z, y))
                y = torch.matmul(y, t)
                z = torch.matmul(t, z)
            return (y * scale.sqrt().view(-1, 1, 1))[:, 0, 0].clamp_min(self.eig_floor ** 0.5)
        # Harden the matrix sqrt without changing the physics. A learned coupling that drifts
        # large can overflow the fp32 matvec -> a non-finite tridiagonal -> eigh emits NaN, and
        # the old clamp_min(NaN)=NaN propagated into a training-killing NaN (clamp floors finite
        # negatives but passes NaN/Inf straight through). Sanitize + symmetrize, then floor the
        # spectrum SMOOTHLY: sqrt(0.5*(x + sqrt(x^2 + eig_floor^2))) equals sqrt(x) for x >> eig_floor
        # (bit-identical in the physical positive-definite region -> accuracy-neutral) and stays
        # finite/kink-free for x <= 0, so an indefinite Ritz value can no longer NaN the forward
        # or its gradient.  eig_floor (1e-8) sits far below the physical spectrum (~omega^2), so
        # this only intervenes in the unphysical near-/sub-zero region.
        tri = torch.nan_to_num(tri, nan=0.0, posinf=0.0, neginf=0.0)
        tri = 0.5 * (tri + tri.transpose(-1, -2))
        # The forward is identical to the previous in-line eigh-quadrature; the custom Function
        # only swaps eigh's degeneracy-singular eigenvector backward for the finite Daleckii-Krein
        # divided-difference gradient (see _SqrtFirstMomentQuad).
        return _SqrtFirstMomentQuad.apply(tri, self.eig_floor, self.eig_ceil)

    def _det_rademacher(self, count: int, m: int, offset: int, *, device, dtype):
        """``count`` deterministic Rademacher vectors over ``m`` atoms x 3 components, probe-indexed
        from ``offset`` (distinct offsets -> distinct, reproducible, makefx-safe vector sets)."""
        p = torch.arange(offset, offset + count, device=device, dtype=dtype).view(count, 1, 1)
        a = torch.arange(m, device=device, dtype=dtype).view(1, m, 1)
        c = torch.arange(3, device=device, dtype=dtype).view(1, 1, 3)
        h = ((p + 1.0) * 12.9898 + (a + 1.0) * 78.233 + (c + 1.0) * 37.719
             + (p + 1.0) * (a + 1.0) * 0.137 + (p + 1.0) * (c + 1.0) * 0.193)
        return torch.where(torch.sin(h) >= 0.0, 1.0, -1.0).to(dtype=dtype)  # [count, m, 3]

    def _lanczos_quad(self, matvec, probes: torch.Tensor, steps: int) -> torch.Tensor:
        """Per-probe Lanczos quadrature estimate of z^T sqrt(C) z. ``probes``: [G, P, m, 3];
        ``matvec`` acts on [G, P, m, 3]. Returns [G, P]. (Extracted from the batched Lanczos so the
        Hutch++ residual can reuse it.)"""
        G, P, m_max, _ = probes.shape
        dim = 3 * m_max
        q = probes.reshape(G * P, dim)
        q_norm = q.norm(dim=-1).clamp_min(1.0e-14)
        q = q / q_norm.view(-1, 1)
        q_prev = torch.zeros_like(q)
        beta_prev = q.new_zeros(G * P)
        alphas: list[torch.Tensor] = []
        betas: list[torch.Tensor] = []
        basis: list[torch.Tensor] = []
        for step in range(steps):
            z = matvec(q.reshape(G, P, m_max, 3)).reshape(G * P, dim)
            z = torch.nan_to_num(z, nan=0.0, posinf=1.0e30, neginf=-1.0e30)
            if step > 0:
                z = z - beta_prev.view(-1, 1) * q_prev
            a = (q * z).sum(dim=-1)
            z = z - a.view(-1, 1) * q
            for old_q in basis:
                z = z - (z * old_q).sum(dim=-1, keepdim=True) * old_q
            b = z.norm(dim=-1)
            alphas.append(a)
            if step + 1 < steps:
                betas.append(b)
            basis.append(q)
            q_prev = q
            q = z / b.clamp_min(1.0e-14).view(-1, 1)
            beta_prev = b
        tri = q.new_zeros(G * P, steps, steps)
        ar = torch.arange(steps, device=q.device)
        tri[:, ar, ar] = torch.stack(alphas, dim=1)
        if steps > 1:
            off = torch.stack(betas, dim=1)
            ar0 = torch.arange(steps - 1, device=q.device)
            tri[:, ar0, ar0 + 1] = off
            tri[:, ar0 + 1, ar0] = off
        return (q_norm.square() * self._sqrt_first_moment(tri)).reshape(G, P)

    def _hutchpp_trace(self, matvec, G: int, m_max: int, steps: int, atom_valid: torch.Tensor,
                       *, device, dtype) -> torch.Tensor:
        """Nystrom-Hutch++ estimate of Tr sqrt(C) per graph [G]. Deflates the dominant eigenspace
        deterministically (zero variance) and Hutchinson-estimates only the residual:
        Tr sqrt(C) = Tr(Q^T sqrt(C) Q) + Tr((I-QQ^T) sqrt(C) (I-QQ^T)).  The first is the Rayleigh-Ritz
        Tr sqrt(Q^T C Q) (one small k x k eigh, NO per-column Lanczos); the second is a few deflated
        Lanczos probes. C is PD here (rescaling guard) so Q^T C Q is PD. Lower variance than vanilla
        Hutchinson => fewer Lanczos probes for the same accuracy."""
        dim = 3 * m_max
        k = max(int(self.hutchpp_sketch), 1)
        mres = max(int(self.hutchpp_residual), 1)
        av = atom_valid.view(G, 1, m_max, 1)
        # sketch the dominant eigenspace: Q = orthonormal(C . S)
        S = self._det_rademacher(k, m_max, 100, device=device, dtype=dtype).unsqueeze(0) * av  # [G,k,m,3]
        Y = matvec(S).reshape(G, k, dim).transpose(1, 2)                      # C.S -> [G, dim, k]
        Y = torch.nan_to_num(Y, nan=0.0, posinf=1.0e30, neginf=-1.0e30)
        Q, _ = torch.linalg.qr(Y)                                             # [G, dim, k]
        # DETACH the randomized sketch basis. torch.linalg.qr's BACKWARD is unstable for
        # rank-deficient / ill-conditioned Y (= C.S for a from-scratch / near-singular C): it solves
        # against R, whose near-zero diagonal blows the 2nd-order (force) gradient to NaN -> the outer
        # optimizer step then corrupts EVERY weight (forward stays finite, so a forward-only nan check
        # never sees it). Q is a Monte-Carlo projection, not the estimated quantity; the trace gradient
        # correctly flows through C in B = Q^T C Q (and the deflation below) with the basis held fixed,
        # which is the standard treatment for stochastic-sketch trace estimators. Forward is unchanged.
        Q = Q.detach()
        # dominant trace via Rayleigh-Ritz: B = Q^T C Q (k x k), Tr sqrt(B)
        CQ = matvec(Q.transpose(1, 2).reshape(G, k, m_max, 3)).reshape(G, k, dim).transpose(1, 2)  # [G,dim,k]
        CQ = torch.nan_to_num(CQ, nan=0.0, posinf=1.0e30, neginf=-1.0e30)     # matvec can overflow fp32 -> sanitize like Y
        B = torch.matmul(Q.transpose(1, 2), CQ)                               # [G, k, k]
        B = 0.5 * (B + B.transpose(1, 2))
        # The k x k Rayleigh-Ritz matrix inherits C's full conditioning (omega spans floor..ceil =>
        # cond up to ~1e11). On a from-scratch / ill-conditioned C the fp32 syevd FAILS TO CONVERGE
        # (LinAlgError "ill-conditioned or too many repeated eigenvalues", code 1) -- eager RAISES and
        # the makefx-compiled path silently emits NaN that kills training. Do this tiny (k x k) eigvalsh
        # in fp64 (tolerates the conditioning) after sanitizing + an eig_floor diagonal jitter (breaks
        # exact repeated eigenvalues); cast back. k == hutchpp_sketch is tiny so fp64 is negligible, and
        # sqrt(clamp(mu)) is bit-identical to the old path in the physical positive region.
        Bd = torch.nan_to_num(B.double(), nan=0.0, posinf=1.0e30, neginf=-1.0e30)
        Bd = Bd + torch.eye(k, device=B.device, dtype=torch.float64).unsqueeze(0) * self.eig_floor
        mu = torch.linalg.eigvalsh(Bd).clamp_min(self.eig_floor).to(B.dtype)  # [G, k]
        tr_dom = mu.sqrt().sum(dim=1)                                         # [G]
        # residual: deflated Hutchinson g' = (I - Q Q^T) g
        g = self._det_rademacher(mres, m_max, 500, device=device, dtype=dtype).unsqueeze(0) * av  # [G,mres,m,3]
        gf = g.reshape(G, mres, dim)
        gdef = (gf - torch.matmul(torch.matmul(gf, Q), Q.transpose(1, 2))).reshape(G, mres, m_max, 3)
        tr_res = self._lanczos_quad(matvec, gdef, steps).mean(dim=1)          # [G]
        return tr_dom + tr_res

    def _forward_looped(
        self,
        node_feats: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_vec: torch.Tensor,
        num_graphs: int | None = None,
        pos: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
        l2_feats: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_atoms = node_feats.shape[0]
        per_atom = node_feats.new_zeros(n_atoms)
        if n_atoms == 0:
            return per_atom.unsqueeze(-1)

        alpha = F.softplus(self.alpha_head(node_feats)).squeeze(-1) + self.alpha_floor
        omega = F.softplus(self.omega_head(node_feats)).squeeze(-1) + self.omega_floor
        # per-atom 3x3 coupling factor W = omega * alpha^{1/2}.  Isotropic: W = omega*sqrt(alpha)*I.
        # Anisotropic: W carries the l=2 polarizability tensor -> off-diagonal block W_i T_ij W_j.
        W_factor = omega.view(-1, 1, 1) * self.polarizability_factor(node_feats, l2_feats)  # [N,3,3]
        beta = F.softplus(self.beta_raw) + 1.0e-6
        coupling_scale = self.coupling_scale
        eye3 = torch.eye(3, dtype=node_feats.dtype, device=node_feats.device)

        pme_fft = self.operator_backend == "pme_fft"
        num_graphs = int(num_graphs) if num_graphs is not None else (int(batch.max().item()) + 1 if batch.numel() else 0)
        pme_kernels = None
        if pme_fft and cell is not None and num_graphs > 1:
            pme_kernels = build_periodic_dipole_pme_kernel_batched(
                cell=cell[:num_graphs],
                mesh_size=int(self.pme_mesh_size),
                assignment=self.pme_assignment,
                device=node_feats.device,
                dtype=node_feats.dtype,
                k_norm_floor=self.pme_k_norm_floor,
                assignment_window_floor=self.pme_assignment_window_floor,
                ewald_alpha_prefactor=self.pme_ewald_alpha_prefactor,
            )
        for g in range(num_graphs):
            if num_graphs == 1:
                idx = torch.arange(n_atoms, dtype=torch.long, device=node_feats.device)
                m = n_atoms
                local = None
                same_graph = None
            else:
                idx = (batch == g).nonzero(as_tuple=True)[0]
                m = idx.size(0)
                if _size_leq_zero(m):
                    continue
                if pme_fft:
                    local = None
                    same_graph = None
                else:
                    local = torch.full((n_atoms,), -1, dtype=torch.long, device=node_feats.device)
                    local[idx] = torch.arange(m, dtype=torch.long, device=node_feats.device)
                    same_graph = (batch[edge_src] == g) & (batch[edge_dst] == g)
            steps = max(int(self.lanczos_steps), 1)
            if self.probe_mode == "basis":
                steps = min(steps, 3 * int(m))

            if pme_fft:
                li = torch.zeros(0, dtype=torch.long, device=node_feats.device)
                lj = torch.zeros(0, dtype=torch.long, device=node_feats.device)
                blocks = node_feats.new_zeros(0, 3, 3)
            else:
                if same_graph is None:
                    es = edge_src
                    ed = edge_dst
                    ev = edge_vec
                    coupling_weight = _canonical_undirected_edge_mask(edge_src, edge_dst, edge_vec).to(dtype=node_feats.dtype)
                    if edge_weight is not None:
                        coupling_weight = coupling_weight * edge_weight.to(dtype=node_feats.dtype)
                    li = ed
                    lj = es
                else:
                    es = edge_src
                    ed = edge_dst
                    ev = edge_vec
                    coupling_weight = (
                        same_graph & _canonical_undirected_edge_mask(edge_src, edge_dst, edge_vec)
                    ).to(dtype=node_feats.dtype)
                    if edge_weight is not None:
                        coupling_weight = coupling_weight * edge_weight.to(dtype=node_feats.dtype)
                    # Keep the per-graph edge tensor shape static for make_fx.  Invalid
                    # cross-graph edges are mapped to atom 0 but receive zero weight.
                    li = local[ed].clamp_min(0)
                    lj = local[es].clamp_min(0)
                r = ev.norm(dim=-1).clamp_min(1.0e-6)
                rhat = ev / r.unsqueeze(-1)
                tensor = (3.0 * rhat.unsqueeze(-1) * rhat.unsqueeze(-2) - eye3) / r.pow(3).view(-1, 1, 1)
                radius = alpha[es].pow(1.0 / 3.0) + alpha[ed].pow(1.0 / 3.0) + 1.0e-6
                damp = 1.0 - torch.exp(-((r / (beta * radius)).clamp_min(0.0)).pow(6))
                scal = coupling_scale * damp
                if coupling_weight is not None:
                    scal = scal * coupling_weight
                # off-diagonal block_ij = coupling_scale * damp * W_{ed} T_ij W_{es}  (W = omega*alpha^{1/2};
                # reduces EXACTLY to the scalar pref*T when isotropic W = omega*sqrt(alpha)*I). The matvec
                # maps v[es] -> contribution at ed, so the block sandwiches T between the two atoms' factors.
                blocks = scal.view(-1, 1, 1) * torch.matmul(torch.matmul(W_factor[ed], tensor), W_factor[es])

            omega_local = omega[idx]
            pme_kernel_g = None
            if pme_kernels is not None:
                pme_kernel_g = (
                    pme_kernels[0][g],
                    pme_kernels[1][g],
                    pme_kernels[2][g],
                )
            matvec = self._build_matvec(
                omega_local=omega_local,
                alpha_local=alpha[idx],
                li=li,
                lj=lj,
                blocks=blocks,
                pos_local=None if pos is None else pos.index_select(0, idx),
                cell=None if cell is None else cell[g],
                coupling_scale=coupling_scale,
                pme_kernel=pme_kernel_g,
                w_local=W_factor.index_select(0, idx),
            )

            probes = self._make_probes(m, device=node_feats.device, dtype=node_feats.dtype)
            n_probe = probes.size(0)
            # --- PD spectral guard: scale the coupling so C = Omega^2 + coupling stays strictly
            # positive-definite (rho of the normalized coupling < 1 - pd_margin). s == 1 in the
            # healthy regime -> operator (and physics) unchanged; only the catastrophe is screened.
            if self.pd_rescale and n_probe > 0:
                rho = self._pd_rho(matvec, omega_local.view(1, -1, 1), probes)
                s_pd = ((1.0 - self.pd_margin) / rho.max().clamp_min(1.0e-6)).clamp(max=1.0).detach()
                _base_mv = matvec
                _omega_sq_d = omega_local.square().view(1, -1, 1)
                def matvec(v, _b=_base_mv, _s=s_pd, _o=_omega_sq_d):
                    return _s * _b(v) + (1.0 - _s) * (_o * v)
            q = probes.reshape(n_probe, -1)
            dim = q.size(1)
            q_norm = q.norm(dim=-1).clamp_min(1.0e-14)
            q = q / q_norm.view(-1, 1)
            q_prev = torch.zeros_like(q)
            beta_prev = q.new_zeros(n_probe)
            alphas: list[torch.Tensor] = []
            betas: list[torch.Tensor] = []
            basis: list[torch.Tensor] = []
            for step in range(steps):
                z = matvec(q.reshape(n_probe, m, 3)).reshape(n_probe, 3 * m)
                # Guard the Krylov recurrence against an overflowing matvec (a large learned
                # coupling can blow up in fp32): cap non-finite entries so a single bad step
                # cannot poison q -- and every later step's q and gradient -- via z/||z||.
                # Identity for finite z, so it is accuracy-neutral on healthy configurations.
                z = torch.nan_to_num(z, nan=0.0, posinf=1.0e30, neginf=-1.0e30)
                if step > 0:
                    z = z - beta_prev.view(-1, 1) * q_prev
                a = (q * z).sum(dim=-1)
                z = z - a.view(-1, 1) * q
                # Modified Gram-Schmidt is cheap here and keeps the tiny Lanczos matrices stable
                # enough for differentiable training smoke tests.
                for old_q in basis:
                    z = z - (z * old_q).sum(dim=-1, keepdim=True) * old_q
                b = z.norm(dim=-1)
                alphas.append(a)
                if step + 1 < steps:
                    betas.append(b)
                basis.append(q)
                q_prev = q
                q = z / b.clamp_min(1.0e-14).view(-1, 1)
                beta_prev = b

            tri = q.new_zeros(n_probe, steps, steps)
            diag = torch.stack(alphas, dim=1)
            ar = torch.arange(steps, device=node_feats.device)
            tri[:, ar, ar] = diag
            if steps > 1:
                off = torch.stack(betas, dim=1)
                ar0 = torch.arange(steps - 1, device=node_feats.device)
                tri[:, ar0, ar0 + 1] = off
                tri[:, ar0 + 1, ar0] = off
            estimates = q_norm.square() * self._sqrt_first_moment(tri)
            if self.probe_mode == "basis":
                trace_sqrt = estimates.sum()
            elif self.probe_mode == "atom-rademacher":
                trace_sqrt = estimates.reshape(max(int(self.num_probes), 1), 3).sum(dim=1).mean()
            else:
                trace_sqrt = estimates.mean()
            e_graph = 0.5 * trace_sqrt - 1.5 * omega_local.sum()
            per_atom[idx] = e_graph / m
        return per_atom.unsqueeze(-1)

    def forward(
        self,
        node_feats: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_vec: torch.Tensor,
        num_graphs: int | None = None,
        pos: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
        l2_feats: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The batched path is numerically equivalent to ``_forward_looped`` for the default
        # ``edge_sparse`` operator with deterministic Rademacher probes -- it runs ONE Lanczos
        # over the (G, P) batch (graphs padded to the max graph/edge size) instead of G sequential
        # per-graph Lanczos runs, which keeps the GPU busy and removes the O(G.E) edge-mask redundancy.
        # ``pme_fft`` (reciprocal matvec) and the non-Rademacher probe modes ("basis" has a
        # variable, m-dependent probe count; "atom-rademacher" a different reduction) keep the proven
        # per-graph loop -- equivalence/utilization there is out of scope for this batching change.
        if self.operator_backend != "edge_sparse" or self.probe_mode != "rademacher":
            per_atom = self._forward_looped(
                node_feats, batch, edge_src, edge_dst, edge_vec,
                num_graphs=num_graphs, pos=pos, cell=cell, l2_feats=l2_feats, edge_weight=edge_weight,
            )
        else:
            per_atom = self._forward_batched(
                node_feats, batch, edge_src, edge_dst, edge_vec,
                num_graphs=num_graphs, pos=pos, cell=cell, l2_feats=l2_feats, edge_weight=edge_weight,
            )
        # Scheduled gentle-start (see __init__): force energy_scale 0->1 over the first
        # `_mbd_ramp_steps` training forwards, then full strength; eval/inference => full strength.
        # ForceTrainer completes this ramp before make_fx tracing, so compiled training traces
        # full-strength MBD instead of baking in the first-step ramp factor.
        if self.training and self._mbd_ramp_count < self._mbd_ramp_steps:
            self._mbd_ramp_count += 1
            ramp = self._mbd_ramp_count / self._mbd_ramp_steps
        else:
            ramp = 1.0
        return ramp * self.energy_scale * per_atom

    def _forward_single_graph_dynamic(
        self,
        node_feats: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_vec: torch.Tensor,
        l2_feats: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single-graph edge-sparse SLQ path that does not bake E into traced buffers."""
        device = node_feats.device
        dtype = node_feats.dtype
        n_atoms = node_feats.shape[0]
        per_atom = node_feats.new_zeros(n_atoms)
        if n_atoms == 0:
            return per_atom.unsqueeze(-1)

        alpha = F.softplus(self.alpha_head(node_feats)).squeeze(-1) + self.alpha_floor
        omega = F.softplus(self.omega_head(node_feats)).squeeze(-1) + self.omega_floor
        W_factor = omega.view(-1, 1, 1) * self.polarizability_factor(node_feats, l2_feats)
        beta = F.softplus(self.beta_raw) + 1.0e-6
        eye3 = torch.eye(3, dtype=dtype, device=device)

        r = edge_vec.norm(dim=-1).clamp_min(1.0e-6)
        rhat = edge_vec / r.unsqueeze(-1)
        tensor = (3.0 * rhat.unsqueeze(-1) * rhat.unsqueeze(-2) - eye3) / r.pow(3).view(-1, 1, 1)
        radius = alpha[edge_src].pow(1.0 / 3.0) + alpha[edge_dst].pow(1.0 / 3.0) + 1.0e-6
        damp = 1.0 - torch.exp(-((r / (beta * radius)).clamp_min(0.0)).pow(6))
        coupling_weight = _canonical_undirected_edge_mask(edge_src, edge_dst, edge_vec).to(dtype=dtype)
        if edge_weight is not None:
            coupling_weight = coupling_weight * edge_weight.to(dtype=dtype)
        scal = self.coupling_scale * damp * coupling_weight
        blocks = scal.view(-1, 1, 1) * torch.matmul(
            torch.matmul(W_factor[edge_dst], tensor), W_factor[edge_src]
        )  # [E,3,3]

        n_probe = max(int(self.num_probes), 1)
        steps = max(int(self.lanczos_steps), 1)
        probes = self._make_probes(n_atoms, device=device, dtype=dtype).unsqueeze(0)  # [1,P,N,3]
        omega_batch = omega.view(1, -1)
        omega_sq = omega_batch.square().view(1, 1, -1, 1)

        def matvec(v: torch.Tensor) -> torch.Tensor:
            # v: [1,P,N,3].  Use the current edge list directly; no [E_trace]
            # padded buffer survives into the TorchScript graph.
            vp = v.squeeze(0)
            y = (omega_sq * v).squeeze(0)
            v_s = vp.index_select(1, edge_src)
            v_d = vp.index_select(1, edge_dst)
            contrib_i = torch.matmul(blocks.unsqueeze(0), v_s.unsqueeze(-1)).squeeze(-1)
            contrib_j = torch.matmul(blocks.transpose(-1, -2).unsqueeze(0), v_d.unsqueeze(-1)).squeeze(-1)
            y = y.index_add(1, edge_dst, contrib_i)
            y = y.index_add(1, edge_src, contrib_j)
            return y.unsqueeze(0)

        if self.pd_rescale and n_probe > 0:
            rho = self._pd_rho(matvec, omega_batch.view(1, 1, -1, 1), probes)
            s_pd = ((1.0 - self.pd_margin) / rho.amax(dim=1).clamp_min(1.0e-6)).clamp(max=1.0).detach()
            base_matvec = matvec
            s_b = s_pd.view(1, 1, 1, 1)

            def matvec(v: torch.Tensor, _base=base_matvec, _s=s_b, _omega_sq=omega_sq) -> torch.Tensor:
                return _s * _base(v) + (1.0 - _s) * (_omega_sq * v)

        trace_sqrt = self._lanczos_quad(matvec, probes, steps).mean(dim=1)  # [1]
        e_graph = 0.5 * trace_sqrt.squeeze(0) - 1.5 * omega.sum()
        atom_count = torch.ones_like(omega).sum().clamp_min(1.0)
        per_atom = torch.ones_like(omega) * (e_graph / atom_count)
        return per_atom.unsqueeze(-1)

    def _forward_batched(
        self,
        node_feats: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_vec: torch.Tensor,
        num_graphs: int | None = None,
        pos: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
        l2_feats: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = node_feats.device
        dtype = node_feats.dtype
        n_atoms = node_feats.shape[0]
        per_atom = node_feats.new_zeros(n_atoms)
        if n_atoms == 0:
            return per_atom.unsqueeze(-1)

        alpha = F.softplus(self.alpha_head(node_feats)).squeeze(-1) + self.alpha_floor
        omega = F.softplus(self.omega_head(node_feats)).squeeze(-1) + self.omega_floor
        W_factor = omega.view(-1, 1, 1) * self.polarizability_factor(node_feats, l2_feats)  # [N,3,3]
        beta = F.softplus(self.beta_raw) + 1.0e-6
        coupling_scale = self.coupling_scale
        eye3 = torch.eye(3, dtype=dtype, device=device)

        num_graphs = int(num_graphs) if num_graphs is not None else (
            int(batch.max().item()) + 1 if batch.numel() else 0
        )
        if num_graphs <= 0:
            return per_atom.unsqueeze(-1)
        if num_graphs == 1 and not self.mbd_static_batch:
            return self._forward_single_graph_dynamic(
                node_feats,
                edge_src,
                edge_dst,
                edge_vec,
                l2_feats=l2_feats,
                edge_weight=edge_weight,
            )

        # --- precompute the per-edge off-diagonal block once for ALL edges (identical to the loop) ---
        # blocks_full[e] = coupling_scale * damp_e * canonical_mask_e * W_{dst} T_e W_{src}; the canonical
        # mask + damping reproduce ``scal`` from the per-graph loop, so a graph just gathers its own rows.
        es_all = edge_src
        ed_all = edge_dst
        r = edge_vec.norm(dim=-1).clamp_min(1.0e-6)
        rhat = edge_vec / r.unsqueeze(-1)
        tensor = (3.0 * rhat.unsqueeze(-1) * rhat.unsqueeze(-2) - eye3) / r.pow(3).view(-1, 1, 1)
        radius = alpha[es_all].pow(1.0 / 3.0) + alpha[ed_all].pow(1.0 / 3.0) + 1.0e-6
        damp = 1.0 - torch.exp(-((r / (beta * radius)).clamp_min(0.0)).pow(6))
        coupling_weight = _canonical_undirected_edge_mask(edge_src, edge_dst, edge_vec).to(dtype=dtype)
        if edge_weight is not None:
            coupling_weight = coupling_weight * edge_weight.to(dtype=dtype)
        scal = coupling_scale * damp * coupling_weight
        blocks_full = scal.view(-1, 1, 1) * torch.matmul(
            torch.matmul(W_factor[ed_all], tensor), W_factor[es_all]
        )  # [E,3,3]

        n_probe = max(int(self.num_probes), 1)  # rademacher: fixed P, identical per graph
        steps = max(int(self.lanczos_steps), 1)
        if self.mbd_static_batch and num_graphs > 1:
            # --- make_fx STATIC path: the bucketing pads every sample to a fixed size, so the G
            # graphs in a batch are EQUAL-SIZE (m_max = n_atoms // num_graphs) and contiguous. The
            # per-graph split is then a RESHAPE (no .nonzero / int(size) -> no data-dependent symint
            # u0, so make_fx can trace it). Local atom id within graph g = global id - g*m_max;
            # padding atoms (if any) carry omega=0 and contribute nothing. ---
            # make_fx STATIC path. Graphs are NOT guaranteed equal-size, so SCATTER each atom to its
            # (graph, within-graph-rank) slot in a [G, m_max] grid with m_max = n_atoms (a safe upper
            # bound -> no overflow). All ops are fixed-shape / make_fx-safe (bincount/cumsum/scatter,
            # no .nonzero, no int(size)). Atoms are contiguous per graph (the loader concatenates), so
            # within-rank = global_id - graph_start. The matvec below stays O(E) on the edges, so the
            # over-pad (m_max=N) only inflates the cheap diagonal + Lanczos-vector ops, not the edges.
            G = num_graphs
            m_max = n_atoms
            counts = torch.zeros(
                G, dtype=batch.dtype, device=batch.device
            ).scatter_add(0, batch, torch.ones_like(batch))                              # [G] atoms/graph
            offsets = torch.cumsum(counts, 0) - counts                                   # [G] first global id/graph
            within = torch.arange(n_atoms, device=device) - offsets.index_select(0, batch)  # [N] within-graph rank
            flat = batch * m_max + within                                                # [N] slot in flat [G*m_max]
            omega_batch = omega.new_zeros(G * m_max).index_copy(0, flat, omega).reshape(G, m_max)
            atom_valid = omega.new_zeros(G * m_max).index_copy(0, flat, torch.ones_like(omega)).reshape(G, m_max)
            fsrc = batch.index_select(0, es_all) * m_max + within.index_select(0, es_all)    # [E] src slot in flat
            fdst = batch.index_select(0, ed_all) * m_max + within.index_select(0, ed_all)    # [E] dst slot in flat
            idx_list = None
            m_list = None
            active = list(range(G))
        else:
            # --- eager DYNAMIC path: per-graph setup loop (.nonzero -> variable per-graph sizes).
            # Each graph maps its global edges to LOCAL atom ids (0..m_g-1) like ``local[ed]``/``local[es]``.
            idx_list: list[torch.Tensor] = []
            li_list: list[torch.Tensor] = []
            lj_list: list[torch.Tensor] = []
            eidx_list: list[torch.Tensor] = []  # which global edges belong to graph g
            m_list: list[int] = []
            if num_graphs == 1:
                idx_g = torch.arange(n_atoms, dtype=torch.long, device=device)
                idx_list.append(idx_g)
                m_list.append(n_atoms)
                li_list.append(ed_all)
                lj_list.append(es_all)
                eidx_list.append(torch.arange(ed_all.numel(), dtype=torch.long, device=device))
            else:
                local = torch.full((n_atoms,), -1, dtype=torch.long, device=device)
                for g in range(num_graphs):
                    idx_g = (batch == g).nonzero(as_tuple=True)[0]
                    m_g = idx_g.size(0)
                    idx_list.append(idx_g)
                    m_list.append(int(m_g))
                    if _size_leq_zero(m_g):
                        li_list.append(torch.zeros(0, dtype=torch.long, device=device))
                        lj_list.append(torch.zeros(0, dtype=torch.long, device=device))
                        eidx_list.append(torch.zeros(0, dtype=torch.long, device=device))
                        continue
                    local.fill_(-1)
                    local[idx_g] = torch.arange(m_g, dtype=torch.long, device=device)
                    same_graph = (batch[edge_src] == g) & (batch[edge_dst] == g)
                    e_sel = same_graph.nonzero(as_tuple=True)[0]
                    eidx_list.append(e_sel)
                    li_list.append(local[ed_all.index_select(0, e_sel)])
                    lj_list.append(local[es_all.index_select(0, e_sel)])
            active = [g for g in range(num_graphs) if not _size_leq_zero(m_list[g])]
            if len(active) == 0:
                return per_atom.unsqueeze(-1)
            m_max = max(m_list[g] for g in active)
            e_counts = [int(eidx_list[g].numel()) for g in active]
            e_max = max(e_counts) if e_counts else 0
            G = len(active)
            omega_batch = node_feats.new_zeros(G, m_max)          # padded atoms -> omega=0
            li_batch = torch.zeros(G, max(e_max, 1), dtype=torch.long, device=device)
            lj_batch = torch.zeros(G, max(e_max, 1), dtype=torch.long, device=device)
            blocks_batch = node_feats.new_zeros(G, max(e_max, 1), 3, 3)  # padded edges -> zero block
            for bi, g in enumerate(active):
                m_g = m_list[g]
                omega_batch[bi, :m_g] = omega.index_select(0, idx_list[g])
                eg = int(e_counts[bi])
                if eg > 0:
                    li_batch[bi, :eg] = li_list[g]
                    lj_batch[bi, :eg] = lj_list[g]
                    blocks_batch[bi, :eg] = blocks_full.index_select(0, eidx_list[g])
            ar_m = torch.arange(m_max, device=device).view(1, m_max)
            m_g_t = torch.tensor([m_list[g] for g in active], device=device, dtype=torch.long).view(G, 1)
            atom_valid = (ar_m < m_g_t).to(dtype=dtype)  # [G, m_max]
        # rademacher probes are deterministic + identical across graphs of equal m; padding atoms zeroed.
        probes_full = self._make_probes(m_max, device=device, dtype=dtype)  # [P, m_max, 3]
        probes_batch = probes_full.unsqueeze(0) * atom_valid.view(G, 1, m_max, 1)  # [G,P,m_max,3]

        omega_sq = omega_batch.square().view(G, 1, m_max, 1)  # [G,1,m_max,1]
        if self.mbd_static_batch and num_graphs > 1:
            # STATIC matvec: flatten v -> [P, N, 3] (N = G*m_max, contiguous by graph) and scatter the
            # coupling with GLOBAL atom indices (edge_src/edge_dst). This needs no [G, e_max] edge
            # reshape, so it handles the (unequal) per-graph edge counts -> make_fx-safe (fixed E).
            _osq_flat = omega_batch.reshape(1, G * m_max, 1).square()   # [1, G*m_max, 1] (padding slots -> 0)
            _bf = blocks_full   # [E, 3, 3]

            def matvec(v: torch.Tensor) -> torch.Tensor:
                P = v.shape[1]
                vp = v.permute(1, 0, 2, 3).reshape(P, G * m_max, 3)     # [P, G*m_max, 3] (graph-contiguous)
                y = _osq_flat * vp
                v_s = vp.index_select(1, fsrc)                          # [P, E, 3] at src slot
                v_d = vp.index_select(1, fdst)                          # [P, E, 3] at dst slot
                # block @ v[src] and block^T @ v[dst] as explicit 3-term sums (elementwise + broadcast,
                # NO batched-matmul reshape -> no [P*E,3,1] symbolic-product view -> inductor-friendly
                # under dynamic shapes). _bf[:,:,j]=column j, _bf[:,j,:]=row j (= block^T's column j).
                ci = _bf[:, :, 0] * v_s[..., 0:1] + _bf[:, :, 1] * v_s[..., 1:2] + _bf[:, :, 2] * v_s[..., 2:3]
                cj = _bf[:, 0, :] * v_d[..., 0:1] + _bf[:, 1, :] * v_d[..., 1:2] + _bf[:, 2, :] * v_d[..., 2:3]
                y = y.index_add(1, fdst, ci)
                y = y.index_add(1, fsrc, cj)
                return y.reshape(P, G, m_max, 3).permute(1, 0, 2, 3)     # [G, P, m_max, 3]
        else:
            idx_i_b = li_batch.view(G, 1, -1, 1)  # [G,1,E,1] scatter/gather targets (atom dim=2)
            idx_j_b = lj_batch.view(G, 1, -1, 1)
            blocks_g = blocks_batch.unsqueeze(1)                  # [G,1,E,3,3]
            blocks_gT = blocks_batch.transpose(-1, -2).unsqueeze(1)

            def matvec(v: torch.Tensor) -> torch.Tensor:
                # v: [G, P, m_max, 3]; C.v = omega^2 v + scatter(blocks @ v[lj]) + scatter(blocks^T @ v[li]).
                # Expand the edge index to v's probe count P so ONE construction serves any P (Lanczos
                # probes, the Hutch++ sketch k, and the deflated residual m all flow through this operator).
                P = v.shape[1]
                idx_i = idx_i_b.expand(G, P, -1, 3)
                idx_j = idx_j_b.expand(G, P, -1, 3)
                y = omega_sq * v
                v_j = torch.gather(v, 2, idx_j)
                v_i = torch.gather(v, 2, idx_i)
                contrib_i = torch.matmul(blocks_g, v_j.unsqueeze(-1)).squeeze(-1)
                contrib_j = torch.matmul(blocks_gT, v_i.unsqueeze(-1)).squeeze(-1)
                y = y.scatter_add(2, idx_i, contrib_i)
                y = y.scatter_add(2, idx_j, contrib_j)
                return y

        # --- PD spectral guard (per graph): scale each graph's coupling so C_g stays strictly
        # positive-definite. rho(M_g) < 1 - pd_margin => lambda_min(Omega^-1 C_g Omega^-1) >=
        # pd_margin > 0. s_g == 1 where healthy (zero bias); only an indefinite C_g is screened.
        if self.pd_rescale:
            # AMORTIZE (eager only): a cheap block-Gershgorin test skips the 10-matvec power iteration
            # whenever C = Omega^2 + coupling is provably PD -- for the normalized coupling M (zero
            # diagonal), rho(M) <= max_i sum_j ||block_ij||_F/(Omega_i Omega_j); if that < 1 - pd_margin
            # then C is strictly PD and no screening is needed. The skip uses a Python bool, which is
            # NOT make_fx-traceable, so it is gated behind the STATIC self.pd_amortize flag: with
            # pd_amortize=False (the makefx path) the branchless always-power-iteration is traced --
            # clamp(max=1) makes s==1 in the healthy regime = the identical zero-bias result, just
            # without skipping the (now compiled, cheap) power iteration. Reuses the coupling blocks.
            _skip = False
            if self.pd_amortize:
                _enorm = blocks_batch.flatten(-2).norm(dim=-1)                   # [G, E] = ||block_e||_F
                _oi = torch.gather(omega_batch, 1, li_batch).clamp_min(1.0e-6)   # [G, E] Omega at li
                _oj = torch.gather(omega_batch, 1, lj_batch).clamp_min(1.0e-6)   # [G, E] Omega at lj
                _rowsum = omega_batch.new_zeros(G, m_max)
                _rowsum.scatter_add_(1, li_batch, _enorm / _oj)                  # atom li <- ||block||/Omega_j
                _rowsum.scatter_add_(1, lj_batch, _enorm / _oi)                  # atom lj <- ||block||/Omega_i
                _gersh = (_rowsum / omega_batch.clamp_min(1.0e-6)).amax(dim=1)   # [G], >= rho(M_g)
                _skip = bool((_gersh < (1.0 - self.pd_margin)).all())           # all graphs provably PD-safe
            if not _skip:
                rho = self._pd_rho(matvec, omega_batch.view(G, 1, m_max, 1), probes_batch)  # [G, n_probe]
                s_pd = ((1.0 - self.pd_margin) / rho.amax(dim=1).clamp_min(1.0e-6)).clamp(max=1.0).detach()  # [G]
                _base_mv = matvec
                _s_b = s_pd.view(G, 1, 1, 1)
                def matvec(v, _b=_base_mv, _s=_s_b, _o=omega_sq):
                    return _s * _b(v) + (1.0 - _s) * (_o * v)

        # --- trace estimate: vanilla SLQ Hutchinson is the default. Hutch++ is retained only for
        # explicit experiments: its detached sketch basis is not a conservative-force default for
        # the flat, diagonal-dominant MBD spectra seen here.
        if self.trace_estimator == "hutch++":
            trace_sqrt = self._hutchpp_trace(matvec, G, m_max, steps, atom_valid, device=device, dtype=dtype)  # [G]
        else:
            trace_sqrt = self._lanczos_quad(matvec, probes_batch, steps).mean(dim=1)  # [G]
        omega_sum = omega_batch.sum(dim=1)                               # [G] (padding omega=0)
        e_graph = 0.5 * trace_sqrt - 1.5 * omega_sum                     # [G]
        if self.mbd_static_batch and num_graphs > 1:
            # vectorized scatter of per-graph energy back to atoms (make_fx-safe; no Python loop /
            # data-dependent index_put). per_atom[a] = e_graph[graph(a)] / atoms_in_graph(a).
            per_atom = (e_graph / counts.clamp_min(1).to(e_graph.dtype)).index_select(0, batch)  # [N]
        else:
            for bi, g in enumerate(active):
                per_atom[idx_list[g]] = e_graph[bi] / m_list[g]
        return per_atom.unsqueeze(-1)


class LongRangeDispersion(nn.Module):
    """Unified long-range dispersion term.

    This wrapper keeps the model forward independent of the concrete dispersion
    implementation. It exposes the learned pairwise-C6 term, the dense MBD
    oracle, and the matrix-free SLQ approximation through one model interface.
    """

    SUPPORTED_MODES = {"pairwise-c6", "mbd", "mbd-slq"}
    SUPPORTED_NEIGHBOR_METHODS = {"auto", "cell", "bruteforce"}

    def __init__(
        self,
        *,
        feature_dim: int,
        mode: str = "pairwise-c6",
        hidden_dim: int = 32,
        cutoff: float = 8.0,
        pbc: bool = True,
        neighbor_method: str = "auto",
        bruteforce_threshold: int = 1024,
        allow_large_bruteforce_fallback: bool = False,
        slq_num_probes: int = 8,
        slq_lanczos_steps: int = 16,
        max_num_neighbors: int | None = None,
        mbd_operator_backend: str = "edge_sparse",
        mbd_pme_mesh_size: int = 32,
        mbd_pme_assignment: str = "pcs",
        mbd_pme_k_norm_floor: float = 1.0e-6,
        mbd_pme_assignment_window_floor: float = 1.0e-6,
        mbd_pme_ewald_alpha_prefactor: float = 5.0,
        mbd_anisotropic_polarizability: bool = False,
        mbd_learnable_energy_scale: bool = True,
        mbd_alpha_floor: float = 1.0e-4,
        min_cutoff: float = 0.0,
        switch_width: float = 0.5,
    ) -> None:
        super().__init__()
        self.mode = str(mode)
        if self.mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported long-range dispersion mode {self.mode!r}; "
                f"supported modes: {sorted(self.SUPPORTED_MODES)}"
            )
        self.cutoff = float(cutoff)
        self.pbc = bool(pbc)
        self.neighbor_method = str(neighbor_method)
        if self.neighbor_method not in self.SUPPORTED_NEIGHBOR_METHODS:
            raise ValueError(
                f"Unsupported dispersion neighbor-list method {self.neighbor_method!r}; "
                f"supported methods: {sorted(self.SUPPORTED_NEIGHBOR_METHODS)}"
            )
        self.bruteforce_threshold = int(bruteforce_threshold)
        if self.bruteforce_threshold < 0:
            raise ValueError("dispersion bruteforce_threshold must be >= 0")
        self.allow_large_bruteforce_fallback = bool(allow_large_bruteforce_fallback)
        self.slq_num_probes = int(slq_num_probes)
        self.slq_lanczos_steps = int(slq_lanczos_steps)
        self.max_num_neighbors = _normalize_max_num_neighbors(max_num_neighbors)
        self.mbd_operator_backend = str(mbd_operator_backend)
        self.mbd_pme_mesh_size = int(mbd_pme_mesh_size)
        self.mbd_pme_assignment = str(mbd_pme_assignment)
        self.mbd_pme_k_norm_floor = float(mbd_pme_k_norm_floor)
        self.mbd_pme_assignment_window_floor = float(mbd_pme_assignment_window_floor)
        self.mbd_pme_ewald_alpha_prefactor = float(mbd_pme_ewald_alpha_prefactor)
        self.mbd_anisotropic_polarizability = bool(mbd_anisotropic_polarizability)
        self.mbd_learnable_energy_scale = bool(mbd_learnable_energy_scale)
        self.mbd_alpha_floor = float(mbd_alpha_floor)
        self.min_cutoff = float(min_cutoff)
        self.switch_width = float(switch_width)
        if self.min_cutoff < 0.0:
            raise ValueError("dispersion min_cutoff must be >= 0")
        if self.switch_width < 0.0:
            raise ValueError("dispersion switch_width must be >= 0")
        if self.min_cutoff > 0.0 and self.min_cutoff >= self.cutoff:
            raise ValueError(
                f"dispersion min_cutoff ({self.min_cutoff}) must be < cutoff ({self.cutoff})"
            )
        if self.mode == "pairwise-c6":
            self.term = PairwiseDispersion(feature_dim=feature_dim, hidden_dim=hidden_dim)
        elif self.mode == "mbd":
            self.term = ManyBodyDispersion(feature_dim=feature_dim, hidden_dim=hidden_dim)
        elif self.mode == "mbd-slq":
            self.term = ManyBodyDispersionSLQ(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_probes=self.slq_num_probes,
                lanczos_steps=self.slq_lanczos_steps,
                operator_backend=self.mbd_operator_backend,
                pme_mesh_size=self.mbd_pme_mesh_size,
                pme_assignment=self.mbd_pme_assignment,
                pme_k_norm_floor=self.mbd_pme_k_norm_floor,
                pme_assignment_window_floor=self.mbd_pme_assignment_window_floor,
                pme_ewald_alpha_prefactor=self.mbd_pme_ewald_alpha_prefactor,
                anisotropic_polarizability=self.mbd_anisotropic_polarizability,
                learnable_energy_scale=self.mbd_learnable_energy_scale,
                alpha_floor=self.mbd_alpha_floor,
            )
        else:  # pragma: no cover - guarded above; future modes land here explicitly.
            raise ValueError(f"Unsupported long-range dispersion mode {self.mode!r}")

    @property
    def uses_cutoff_neighbor_list(self) -> bool:
        return dispersion_mode_uses_cutoff_edges(
            self.mode,
            mbd_operator_backend=getattr(self.term, "operator_backend", self.mbd_operator_backend),
        )

    @property
    def uses_canonical_undirected_edges(self) -> bool:
        return dispersion_mode_uses_canonical_edges(self.mode)

    def exports_mbd_source(self) -> bool:
        """True when this dispersion term can emit a per-atom (omega, alpha) source for the C++ MBD
        backend (the mbd-slq head). The model defers the coupled-dipole energy to C++ when it emits."""
        return self.mode == "mbd-slq" and hasattr(self.term, "emit_source")

    def emit_source(self, node_feats: torch.Tensor, l2_feats: torch.Tensor | None = None) -> torch.Tensor:
        """Deploy: per-atom MBD source from the SLQ head for the C++ solver. Isotropic [N,2]=(omega,alpha);
        anisotropic [N,8]=(omega, alpha_iso, 6 components of B=alpha^{1/2}) when l2_feats is supplied."""
        return self.term.emit_source(node_feats, l2_feats)

    def mbd_beta(self) -> float:
        return float(self.term.mbd_beta())

    def mbd_coupling_scale(self) -> float:
        return float(self.term.mbd_coupling_scale())

    def mbd_energy_scale(self) -> float:
        return float(self.term.mbd_energy_scale()) if hasattr(self.term, "mbd_energy_scale") else 1.0

    def forward(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        *,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_lengths: torch.Tensor,
        edge_vec: torch.Tensor | None = None,
        cutoff: float | None = None,
        pbc: bool | None = None,
        l2_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cutoff_value = self.cutoff if cutoff is None else float(cutoff)
        pbc_value = self.pbc if pbc is None else bool(pbc)
        cutoff_edges = self.uses_cutoff_neighbor_list
        if not cutoff_edges:
            d_src = torch.zeros(0, dtype=torch.long, device=pos.device)
            d_dst = torch.zeros(0, dtype=torch.long, device=pos.device)
            d_len = pos.new_zeros(0)
            d_vec = pos.new_zeros(0, 3)
        elif cutoff_value and cutoff_value > 0.0:
            d_src, d_dst, d_shift = dispersion_neighbor_list(
                pos,
                batch,
                cell,
                cutoff_value,
                pbc=pbc_value,
                canonical_undirected=self.uses_canonical_undirected_edges,
                method=self.neighbor_method,
                bruteforce_threshold=self.bruteforce_threshold,
                max_num_neighbors=self.max_num_neighbors,
                allow_large_bruteforce_fallback=self.allow_large_bruteforce_fallback,
            )
            shift_vecs = torch.einsum("ni,nij->nj", d_shift.to(pos.dtype), cell[batch[d_dst]])
            d_vec = pos[d_dst] - pos[d_src] + shift_vecs
            d_len = d_vec.norm(dim=1)
        else:
            d_src, d_dst, d_len, d_vec = edge_src, edge_dst, edge_lengths, edge_vec

        edge_weight = None
        if cutoff_edges and d_len.numel():
            window_cutoff = cutoff_value if cutoff_value and cutoff_value > 0.0 else self.cutoff
            # C2 smooth windows replace the old hard min_cutoff deletion and the hard outer
            # cutoff. Short-range pairs still get zero long-range weight, but without an
            # energy/force jump at either boundary.
            edge_weight = _dispersion_edge_window(
                d_len,
                cutoff=window_cutoff,
                min_cutoff=self.min_cutoff,
                switch_width=self.switch_width,
            )

        if self.mode == "pairwise-c6":
            return self.term(node_feats, d_src, d_dst, d_len, edge_weight=edge_weight)
        if self.mode == "mbd":
            if d_vec is None:
                raise ValueError("MBD dispersion requires edge_vec or cutoff-based neighbor construction")
            return self.term(node_feats, batch, d_src, d_dst, d_vec, num_graphs=int(cell.shape[0]), edge_weight=edge_weight)
        if self.mode == "mbd-slq":
            if d_vec is None and cutoff_edges:
                raise ValueError("MBD dispersion requires edge_vec or cutoff-based neighbor construction")
            return self.term(
                node_feats,
                batch,
                d_src,
                d_dst,
                d_vec,
                num_graphs=int(cell.shape[0]),
                pos=pos,
                cell=cell,
                l2_feats=l2_feats,
                edge_weight=edge_weight,
            )
        raise ValueError(f"Unsupported long-range dispersion mode {self.mode!r}")


def normalize_dispersion_mode(
    *,
    long_range_dispersion: bool = False,
    long_range_dispersion_mode: str | None = None,
) -> str:
    """Resolve legacy boolean and explicit mode into one stable mode string."""

    if long_range_dispersion_mode is None:
        return "pairwise-c6" if bool(long_range_dispersion) else "none"
    mode = str(long_range_dispersion_mode)
    if mode == "none" and bool(long_range_dispersion):
        return "pairwise-c6"
    return mode


def dispersion_mode_uses_canonical_edges(mode: str | None) -> bool:
    """Whether the dispersion edge convention is one undirected MBD representative."""
    return str(mode or "none") in {"mbd", "mbd-slq"}


def dispersion_mode_uses_cutoff_edges(
    mode: str | None,
    *,
    mbd_operator_backend: str = "edge_sparse",
) -> bool:
    """Whether model/training should build or consume a cutoff dispersion graph.

    The reciprocal-space MBD prototype applies the dipole-tensor operator through
    FFT matvecs, so it intentionally bypasses the real-space dispersion graph.
    """
    mode_s = str(mode or "none")
    if mode_s == "mbd-slq" and str(mbd_operator_backend) == "pme_fft":
        return False
    return mode_s in {"pairwise-c6", "mbd", "mbd-slq"}


def dispersion_mode_needs_deployment_edges(
    mode: str | None,
    *,
    mbd_operator_backend: str = "edge_sparse",
) -> bool:
    """Whether mff/torch export needs a second explicit dispersion neighbor list."""
    mode_s = str(mode or "none")
    if not dispersion_mode_uses_cutoff_edges(mode_s, mbd_operator_backend=mbd_operator_backend):
        return False
    return mode_s in {"mbd", "mbd-slq"}


def dispersion_deployment_graph_rule(
    mode: str | None,
    *,
    mbd_operator_backend: str = "edge_sparse",
) -> str:
    """Stable metadata label for the deployment dispersion graph convention."""
    mode_s = str(mode or "none")
    backend_s = str(mbd_operator_backend)
    if mode_s == "none":
        return "none"
    if mode_s == "pairwise-c6":
        return "main_neighbor_graph"
    if mode_s == "mbd-slq" and backend_s == "pme_fft":
        return "pme_fft_matvec_prototype"
    if mode_s in {"mbd", "mbd-slq"}:
        return "explicit_canonical_single_image_edge_sparse"
    return "unknown"


def dispersion_training_graph_rule(
    mode: str | None,
    *,
    mbd_operator_backend: str = "edge_sparse",
) -> str:
    """Stable metadata label for the training-time dispersion graph convention."""
    mode_s = str(mode or "none")
    backend_s = str(mbd_operator_backend)
    if mode_s == "none":
        return "none"
    if mode_s == "pairwise-c6":
        return "directed_cutoff_or_main_neighbor_graph"
    if mode_s == "mbd-slq" and backend_s == "pme_fft":
        return "pme_fft_matvec_no_cutoff_edges"
    if mode_s in {"mbd", "mbd-slq"}:
        return "explicit_or_built_canonical_cutoff_edge_sparse"
    return "unknown"


def dispersion_train_deploy_graph_compatibility(
    mode: str | None,
    *,
    mbd_operator_backend: str = "edge_sparse",
) -> str:
    """Stable label for how training dispersion edges relate to deployment.

    This is intentionally mode/backend-level metadata, not a promise about a
    particular training cell. Edge-sparse MBD can train on exact multi-image
    small-cell graphs, while current mff/torch deployment only accepts the
    single-image canonical graph and validates that condition at runtime.
    """
    mode_s = str(mode or "none")
    backend_s = str(mbd_operator_backend)
    if mode_s == "none":
        return "none"
    if mode_s == "pairwise-c6":
        return "shared_main_neighbor_graph"
    if mode_s == "mbd-slq" and backend_s == "pme_fft":
        return "matched_pme_fft_matvec"
    if mode_s in {"mbd", "mbd-slq"}:
        return "conditional_on_single_image_cutoff"
    return "unknown"


def build_long_range_dispersion(
    *,
    mode: str,
    feature_dim: int,
    hidden_dim: int = 32,
    cutoff: float = 8.0,
    pbc: bool = True,
    neighbor_method: str = "auto",
    bruteforce_threshold: int = 1024,
    allow_large_bruteforce_fallback: bool = False,
    slq_num_probes: int = 8,
    slq_lanczos_steps: int = 16,
    max_num_neighbors: int | None = None,
    mbd_operator_backend: str = "edge_sparse",
    mbd_pme_mesh_size: int = 32,
    mbd_pme_assignment: str = "pcs",
    mbd_pme_k_norm_floor: float = 1.0e-6,
    mbd_pme_assignment_window_floor: float = 1.0e-6,
    mbd_pme_ewald_alpha_prefactor: float = 5.0,
    mbd_anisotropic_polarizability: bool = False,
    mbd_learnable_energy_scale: bool = True,
    mbd_alpha_floor: float = 1.0e-4,
    dispersion_min_cutoff: float = 0.0,
    dispersion_switch_width: float = 0.5,
) -> LongRangeDispersion | None:
    mode = str(mode)
    if mode == "none":
        return None
    return LongRangeDispersion(
        feature_dim=feature_dim,
        mode=mode,
        hidden_dim=hidden_dim,
        cutoff=cutoff,
        pbc=pbc,
        neighbor_method=neighbor_method,
        bruteforce_threshold=bruteforce_threshold,
        allow_large_bruteforce_fallback=allow_large_bruteforce_fallback,
        slq_num_probes=slq_num_probes,
        slq_lanczos_steps=slq_lanczos_steps,
        max_num_neighbors=max_num_neighbors,
        mbd_operator_backend=mbd_operator_backend,
        mbd_pme_mesh_size=mbd_pme_mesh_size,
        mbd_pme_assignment=mbd_pme_assignment,
        mbd_pme_k_norm_floor=mbd_pme_k_norm_floor,
        mbd_pme_assignment_window_floor=mbd_pme_assignment_window_floor,
        mbd_pme_ewald_alpha_prefactor=mbd_pme_ewald_alpha_prefactor,
        mbd_anisotropic_polarizability=mbd_anisotropic_polarizability,
        mbd_learnable_energy_scale=mbd_learnable_energy_scale,
        mbd_alpha_floor=mbd_alpha_floor,
        min_cutoff=dispersion_min_cutoff,
        switch_width=dispersion_switch_width,
    )
