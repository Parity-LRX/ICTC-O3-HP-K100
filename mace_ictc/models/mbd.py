"""Many-body dispersion (MBD) building blocks: the dipole-dipole field operator via the shared
ReciprocalBackend (Ewald split), and -- later -- the coupled-dipole matvec + SLQ spectral solver.

The dipole field E_i = sum_j T_ij . mu_j with T_ab = d_a d_b (1/r) (the rank-2 dipole-dipole tensor)
is split Ewald-style into:

    T = T_SR  (real-space, erfc-damped, over a neighbour list)
      + T_LR  (reciprocal, PME: spread mu -> FFT -> x (-4pi/V k_a k_b/k^2 e^{-k^2/4a^2}) -> iFFT -> gather)
      - T_self (the exact discrete spread/filter/gather self block)
    tinfoil boundary: drop k=0.

This is the MBD-specific TENSOR kernel that rides on the SAME grid backend as the scalar electrostatic
PME. The total field is exactly alpha-independent (the split is exact), which is the correctness test.
"""

from __future__ import annotations

import math

import torch

from mace_ictc.models.long_range import (
    _dipole_pme_kernel_mesh,
    _dipole_pme_self_blocks,
)
from mace_ictc.models.reciprocal_backend import ReciprocalBackend

_SQRT_PI = math.sqrt(math.pi)


def ewald_b_functions(r: torch.Tensor, alpha: float, *, floor: float = 1.0e-12):
    """B_0,B_1,B_2 for the erfc-damped 1/r: B_0=erfc(ar)/r, B_{n+1}=[(2n+1)B_n+(2a^2)^{n+1}/(a*sqrtpi) e^{-a^2 r^2}]/r^2."""
    r = r.clamp_min(floor)
    r2 = r * r
    gauss = torch.exp(-(alpha * alpha) * r2)
    b0 = torch.erfc(alpha * r) / r
    b1 = (b0 + (2.0 * alpha * alpha) / (alpha * _SQRT_PI) * gauss) / r2
    b2 = (3.0 * b1 + ((2.0 * alpha * alpha) ** 2) / (alpha * _SQRT_PI) * gauss) / r2
    return b0, b1, b2


def dipole_field(
    backend: ReciprocalBackend,
    pos: torch.Tensor,
    mu: torch.Tensor,
    cell: torch.Tensor,
    *,
    alpha: float,
    src: torch.Tensor,
    dst: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    """Periodic Ewald dipole field E_i = sum_j T_ij mu_j (tinfoil, k=0 dropped).

    pos [N,3], mu [N,3], cell [3,3]; (src,dst,shifts) a real-space neighbour list (dst<-src with
    integer cell shift) covering the erfc range. Returns E [N,3].
    """
    N = pos.size(0)
    dtype = pos.dtype

    # --- reciprocal T_LR . mu  (PME, 3-channel) ---
    frac = backend.frac(pos, cell)
    k_cart, k_norm, volume = backend.k_grid(cell, dtype=dtype)            # [K,3],[K],scalar
    mu_k = backend.fftn(backend.spread(frac, mu)).reshape(-1, 3)          # [K,3] complex
    k_c = k_cart.to(mu_k.dtype)
    kdotmu = (k_c * mu_k).sum(-1)                                         # [K] complex  (k . mu(k))
    screen = torch.exp(-(k_norm.square()) / (4.0 * alpha * alpha))
    wdeconv = backend.assignment_window(device=pos.device, dtype=dtype)
    spectral = -(4.0 * math.pi) / volume * screen * wdeconv                # T~ = -4pi/V k k/k^2 ...
    spectral = torch.where(k_norm > backend.k_norm_floor, spectral, torch.zeros_like(spectral))
    scale = spectral / k_norm.square().clamp_min(backend.k_norm_floor * backend.k_norm_floor)
    e_k = (scale.to(mu_k.dtype).unsqueeze(-1) * k_c) * kdotmu.unsqueeze(-1)  # [K,3] = scale k_a (k.mu)
    m = backend.mesh_size
    e_mesh = backend.ifftn(e_k.reshape(m, m, m, 3)).real * (float(m) ** 3)
    field = backend.gather(frac, e_mesh)                                  # [N,3]

    # --- exact discrete reciprocal self block: remove the same spread/filter/gather response
    # each atom's smeared dipole applies to itself. This removes both the continuum
    # -(4a^3/3sqrtpi) term and the assignment-dependent grid residual.
    kernel_mesh = _dipole_pme_kernel_mesh(
        mesh_size=m,
        k_cart=k_cart,
        k2=k_norm.square(),
        spectral=spectral,
        k_norm_floor=backend.k_norm_floor,
    )
    self_blocks = _dipole_pme_self_blocks(
        frac,
        mesh_size=m,
        assignment=backend.assignment,
        assignment_offsets=backend.assignment_offsets,
        kernel_mesh_flat=kernel_mesh,
    )
    field = field - torch.matmul(self_blocks, mu.unsqueeze(-1)).squeeze(-1)

    # --- real-space T_SR . mu over the neighbour list (T_ab = -B1 d_ab + B2 r_a r_b) ---
    if src.numel() > 0:
        shift_cart = shifts.to(dtype) @ cell.to(dtype)
        rvec = pos.index_select(0, dst) - pos.index_select(0, src) + shift_cart   # [E,3]  (i<-j)
        r = torch.linalg.vector_norm(rvec, dim=-1)
        b0, b1, b2 = ewald_b_functions(r, alpha)
        mu_src = mu.index_select(0, src)                                  # mu_j  [E,3]
        rdotmu = (rvec * mu_src).sum(-1)                                  # [E]
        contrib = -b1.unsqueeze(-1) * mu_src + b2.unsqueeze(-1) * rvec * rdotmu.unsqueeze(-1)  # T_SR mu_j
        field = field.index_add(0, dst, contrib.to(dtype))
    return field


# --------------------------------------------------------------------------------------------------
# MBD coupled-dipole matvec + Stochastic Lanczos Quadrature (SLQ) for E_MBD = 1/2 Tr[sqrt V] - 3/2 sum w
# --------------------------------------------------------------------------------------------------
def coupled_dipole_matvec(x, omega, alpha, field_fn):
    """C . x for the MBD coupled-dipole matrix C_pq = w_i^2 d_pq + (1-d) w_i w_j sqrt(a_i a_j) T_ij^LR.

    x [N,3]; omega [N]; alpha [N] (screened static polarizability); field_fn(mu)->[N,3] the dipole
    field T.mu (excludes the i=i,n=0 self -> the (1-d) off-diagonal). Returns C.x [N,3].
    """
    wsa = (omega * alpha.clamp_min(0).sqrt()).unsqueeze(-1)  # [N,1]
    return (omega.unsqueeze(-1) ** 2) * x + wsa * field_fn(wsa * x)


def lanczos_decompose(matvec, z, steps: int):
    """Lanczos on (C, z) with full reorthogonalization. Returns (Q [n,k], theta [k], U [k,k], znorm)
    so that f(C) z ~= znorm * Q @ U @ (f(theta) * U[0,:]) and z^T f(C) z ~= znorm^2 sum_j U[0,j]^2 f(theta_j)."""
    n = z.numel()
    znorm = (z * z).sum().sqrt()
    q = z / znorm
    Q = []
    alphas, betas = [], []
    beta = z.new_zeros(())
    q_prev = torch.zeros_like(q)
    for _ in range(min(steps, n)):
        Q.append(q)
        w = matvec(q)
        a = (q * w).sum()
        alphas.append(a)
        w = w - a * q - beta * q_prev
        for qi in Q:  # full reorthogonalization (small m -> cheap, stable)
            w = w - (qi * w).sum() * qi
        beta = (w * w).sum().sqrt()
        betas.append(beta)
        if float(beta) < 1e-12:
            break
        q_prev = q
        q = w / beta
    k = len(alphas)
    T = torch.diag(torch.stack(alphas))
    if k > 1:
        off = torch.stack(betas[: k - 1])
        T = T + torch.diag(off, 1) + torch.diag(off, -1)
    theta, U = torch.linalg.eigh(T)
    return torch.stack(Q, dim=1), theta, U, znorm


def lanczos_sqrt_quadform(matvec, z, steps: int):
    """z^T sqrt(C) z via Lanczos quadrature."""
    _Q, theta, U, znorm = lanczos_decompose(matvec, z, steps)
    return (znorm ** 2) * (U[0, :].square() * theta.clamp_min(0).sqrt()).sum()


def lanczos_apply(matvec, z, steps: int, fn):
    """fn(C) z via Lanczos (matvec-only): Q @ U @ (fn(theta) * (U^T (znorm e_1)))."""
    Q, theta, U, znorm = lanczos_decompose(matvec, z, steps)
    coeff = fn(theta) * (U[0, :] * znorm)  # U^T (znorm e_1) = znorm * U[0,:]
    return Q @ (U @ coeff)


def slq_trace_sqrt(matvec, n: int, probes: torch.Tensor, steps: int):
    """Tr[sqrt C] ~= (1/R) sum_r z_r^T sqrt(C) z_r, Hutchinson over the given Rademacher probes [R,n]."""
    acc = probes.new_zeros(())
    R = probes.size(0)
    for r in range(R):
        acc = acc + lanczos_sqrt_quadform(matvec, probes[r], steps)
    return acc / R


def make_probes(n: int, num_probes: int, *, device, dtype, seed: int = 0) -> torch.Tensor:
    """Fixed Rademacher probes [R,n] (deterministic seed -> conservative, reproducible SLQ surrogate;
    use the IDENTICAL set in training and deployment)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randint(0, 2, (num_probes, n), generator=gen, dtype=torch.int64).to(dtype=dtype, device=device)
    return 2.0 * z - 1.0


def mbd_energy_slq(omega, alpha, field_fn, *, num_probes: int = 64, lanczos_steps: int = 20, seed: int = 0):
    """E_MBD = 1/2 Tr[sqrt C] - 3/2 sum_i omega_i, via SLQ (no eigendecomposition of C). omega,alpha [N]."""
    N = omega.size(0)
    n = 3 * N

    def mv(v):  # [3N] -> [3N]
        return coupled_dipole_matvec(v.view(N, 3), omega, alpha, field_fn).reshape(-1)

    probes = make_probes(n, num_probes, device=omega.device, dtype=omega.dtype, seed=seed)
    tr_sqrt = slq_trace_sqrt(mv, n, probes, lanczos_steps)
    return 0.5 * tr_sqrt - 1.5 * omega.sum()


class _TrSqrtSLQ(torch.autograd.Function):
    """Tr[sqrt C] via SLQ with a CUSTOM, matvec-only backward (no autograd through the Lanczos
    intermediates): grad = 1/2 (1/R) sum_r (C^{-1/2} z_r)^T (dC/dtheta) z_r, where C^{-1/2} z_r is one
    more Lanczos apply. Forward stores only the probes + the R resolvent vectors w_r = C^{-1/2} z_r."""

    @staticmethod
    def forward(ctx, probes, steps, field_fn, N, omega, alpha):
        with torch.no_grad():
            def mv(v):
                return coupled_dipole_matvec(v.view(N, 3), omega, alpha, field_fn).reshape(-1)

            tr = slq_trace_sqrt(mv, 3 * N, probes, steps)
            inv_sqrt = lambda t: t.clamp_min(1e-12).rsqrt()
            W = torch.stack([lanczos_apply(mv, probes[r], steps, inv_sqrt) for r in range(probes.size(0))])
        ctx.save_for_backward(probes, W, omega, alpha)
        ctx.field_fn, ctx.N = field_fn, N
        return tr

    @staticmethod
    def backward(ctx, g):
        probes, W, omega, alpha = ctx.saved_tensors
        N, R = ctx.N, probes.size(0)
        om = omega.detach().requires_grad_(True)
        al = alpha.detach().requires_grad_(True)
        with torch.enable_grad():
            s = om.new_zeros(())
            for r in range(R):
                vz = coupled_dipole_matvec(probes[r].view(N, 3), om, al, ctx.field_fn).reshape(-1)
                s = s + (W[r] * vz).sum()
            go, ga = torch.autograd.grad(s * (0.5 / R), [om, al])
        return None, None, None, None, g * go, g * ga


def mbd_energy_slq_lowmem(omega, alpha, field_fn, *, num_probes=64, lanczos_steps=20, seed=0):
    """E_MBD via SLQ with the custom matvec-only backward (LOW MEMORY: stores only the R resolvent
    vectors, not the Lanczos graph). NOTE: the custom grad uses the trace identity dTr[sqrt C] =
    1/2 Tr[C^{-1/2} dC] -- an unbiased estimator of the TRUE (infinite-probe) gradient, but a
    DIFFERENT estimator from the fixed-probe surrogate's exact gradient, so it is not exactly
    conservative at finite R (~1.5% at R=48, -> 0 as R->inf). For CONSERVATIVE MD forces use
    mbd_energy_slq (autograd = exact surrogate gradient); use this for memory-bound training."""
    N = omega.size(0)
    probes = make_probes(3 * N, num_probes, device=omega.device, dtype=omega.dtype, seed=seed)
    tr = _TrSqrtSLQ.apply(probes, lanczos_steps, field_fn, N, omega, alpha)
    return 0.5 * tr - 1.5 * omega.sum()


# --------------------------------------------------------------------------------------------------
# Chebyshev Tr[sqrt C] -- the DEPLOYMENT spectral solver: pure matvec + fixed-degree polynomial, NO
# eigendecomposition (unlike Lanczos' m x m eigh) -> torch.compile / AOTInductor traceable (static
# control flow). Spectral bounds via matvec-only power iteration (also no eigh).
# --------------------------------------------------------------------------------------------------
def power_iter_lambda_max(matvec, n, *, steps, device, dtype, seed=1):
    """Largest eigenvalue (Rayleigh quotient) via fixed-step power iteration -- matvec-only."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(n, generator=gen, dtype=dtype).to(device)
    v = v / (v * v).sum().sqrt()
    lam = v.new_zeros(())
    for _ in range(steps):
        w = matvec(v)
        lam = (v * w).sum()
        v = w / (w * w).sum().sqrt().clamp_min(1e-30)
    return lam


def chebyshev_coeffs_sqrt(degree: int, lmin, lmax, *, device, dtype):
    """Chebyshev coefficients of sqrt(x) on [lmin,lmax] (Chebyshev-Gauss quadrature). Deterministic,
    fixed degree -> precomputable, static-shape (AOTI-friendly)."""
    M = degree + 1
    j = torch.arange(M, device=device, dtype=dtype) + 0.5
    theta = math.pi * j / M
    x = 0.5 * (lmax + lmin) + 0.5 * (lmax - lmin) * torch.cos(theta)
    fx = x.clamp_min(0).sqrt()
    k = torch.arange(degree + 1, device=device, dtype=dtype)
    c = (2.0 / M) * (fx.unsqueeze(0) * torch.cos(k.unsqueeze(1) * theta.unsqueeze(0))).sum(1)
    c = c.clone()
    c[0] = c[0] * 0.5
    return c  # [degree+1]


def chebyshev_trace_sqrt(matvec, n, probes, degree: int, lmin, lmax):
    """Tr[sqrt C] ~= (1/R) sum_r sum_k c_k z_r^T T_k(C~) z_r via the 3-term recurrence (matvec-only,
    no eigensolve). C~ = (2C - (lmax+lmin)I)/(lmax-lmin) maps the spectrum to [-1,1]."""
    c = chebyshev_coeffs_sqrt(degree, lmin, lmax, device=probes.device, dtype=probes.dtype)
    a = 2.0 / (lmax - lmin)
    b = -(lmax + lmin) / (lmax - lmin)

    def smv(v):
        return a * matvec(v) + b * v

    acc = probes.new_zeros(())
    for r in range(probes.size(0)):
        z = probes[r]
        t_prev = z
        t_cur = smv(z)
        s = c[0] * (z * z).sum() + c[1] * (z * t_cur).sum()
        for k in range(2, degree + 1):
            t_next = 2.0 * smv(t_cur) - t_prev
            s = s + c[k] * (z * t_next).sum()
            t_prev, t_cur = t_cur, t_next
        acc = acc + s
    return acc / probes.size(0)


def mbd_energy_chebyshev(omega, alpha, field_fn, *, num_probes=64, degree=24, pad=0.05, power_steps=20, seed=0):
    """E_MBD via Chebyshev Tr[sqrt C] (deployment / AOTI path: no eigendecomposition). Spectral bounds
    [lmin,lmax] from matvec-only power iteration on C and on (lmax I - C), detached (constants)."""
    N = omega.size(0)
    n = 3 * N

    def mv(v):
        return coupled_dipole_matvec(v.view(N, 3), omega, alpha, field_fn).reshape(-1)

    with torch.no_grad():
        lmax = power_iter_lambda_max(mv, n, steps=power_steps, device=omega.device, dtype=omega.dtype, seed=1)
        lmax = float(lmax) * (1.0 + pad)
        gap = power_iter_lambda_max(lambda v: lmax * v - mv(v), n, steps=power_steps,
                                    device=omega.device, dtype=omega.dtype, seed=2)
        lmin = max((lmax - float(gap)) * (1.0 - pad), 1e-6)  # assume C PD
    probes = make_probes(n, num_probes, device=omega.device, dtype=omega.dtype, seed=seed)
    tr = chebyshev_trace_sqrt(mv, n, probes, degree, lmin, lmax)
    return 0.5 * tr - 1.5 * omega.sum()


def mbd_energy_dense(omega, alpha, field_fn):
    """Reference: build C densely (matvec on the 3N basis), eigh, E = 1/2 sum sqrt(lambda) - 3/2 sum w."""
    N = omega.size(0)
    n = 3 * N
    cols = []
    eye = torch.eye(n, dtype=omega.dtype, device=omega.device)
    for j in range(n):
        cols.append(coupled_dipole_matvec(eye[j].view(N, 3), omega, alpha, field_fn).reshape(-1))
    C = torch.stack(cols, dim=1)
    C = 0.5 * (C + C.T)  # symmetrize (numerical)
    lam = torch.linalg.eigvalsh(C)
    return 0.5 * lam.clamp_min(0).sqrt().sum() - 1.5 * omega.sum()
