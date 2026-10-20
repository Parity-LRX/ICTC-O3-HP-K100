// Task 5/5: C++ MBD solver -- mirrors the validated Python (mace_ictc/models/mbd.py +
// reciprocal_backend.py) 1:1 using the libtorch C++ API. Shares the PME/cuFFT grid ops with the
// scalar electrostatics (self-contained B-spline spread/gather matching the Python PME assignment
// for parity). Chebyshev Tr[sqrt C] (no eigensolve) for the deployment hot path.
//
// Build the standalone parity test (compares C++ mbd_energy to a Python reference):
//   g++ -O2 -std=c++17 -DMBD_STANDALONE_TEST mff_mbd_solver.cpp \
//       -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
//       -L$TORCH/lib -ltorch -ltorch_cpu -lc10 -o mbd_test && ./mbd_test
#include "mff_mbd_solver.h"

#include <algorithm>
#include <cmath>
#include <dlfcn.h>

namespace mfftorch {

struct OptionalGilRelease {
  using Py_IsInitialized_Fn = int (*)();
  using PyGILState_Check_Fn = int (*)();
  using PyEval_SaveThread_Fn = void* (*)();
  using PyEval_RestoreThread_Fn = void (*)(void*);

  void* thread_state = nullptr;
  PyEval_RestoreThread_Fn restore = nullptr;

  OptionalGilRelease() {
    auto py_is_initialized = reinterpret_cast<Py_IsInitialized_Fn>(dlsym(RTLD_DEFAULT, "Py_IsInitialized"));
    auto py_gil_check = reinterpret_cast<PyGILState_Check_Fn>(dlsym(RTLD_DEFAULT, "PyGILState_Check"));
    auto py_eval_save = reinterpret_cast<PyEval_SaveThread_Fn>(dlsym(RTLD_DEFAULT, "PyEval_SaveThread"));
    auto py_eval_restore =
        reinterpret_cast<PyEval_RestoreThread_Fn>(dlsym(RTLD_DEFAULT, "PyEval_RestoreThread"));
    if (!py_is_initialized || !py_gil_check || !py_eval_save || !py_eval_restore) return;
    if (!py_is_initialized()) return;
    if (!py_gil_check()) return;
    thread_state = py_eval_save();
    restore = py_eval_restore;
  }

  ~OptionalGilRelease() {
    if (thread_state && restore) restore(thread_state);
  }
};

// integer FFT frequencies [-floor(M/2)..ceil(M/2)-1] in the torch.fft order (0..M/2-1, -M/2..-1).
static torch::Tensor fft_freqs(int M, torch::TensorOptions opt) {
  auto f = torch::arange(M, opt);
  auto split = (M + 1) / 2;  // first 'split' are >=0
  return torch::where(f < split, f, f - M);
}

// stencil size + window exponent per assignment (matches Python _assignment_stencil_size).
static int assignment_stencil(int a) { return a == 1 ? 4 : 2; }  // 1=pcs (4-pt cubic), 0=cic (2-pt)

static double resolve_dispersion_switch_width(double cutoff, double min_cutoff, double requested_width) {
  if (requested_width <= 0.0) return 0.0;
  if (cutoff <= 0.0) return min_cutoff > 0.0 ? requested_width : 0.0;
  const double span = cutoff - std::max(min_cutoff, 0.0);
  if (span <= 0.0) return 0.0;
  return std::max(0.0, std::min(requested_width, 0.5 * span));
}

static torch::Tensor quintic_smoothstep(const torch::Tensor& x) {
  auto t = x.clamp(0.0, 1.0);
  return t * t * t * (10.0 + t * (-15.0 + 6.0 * t));
}

static torch::Tensor dispersion_edge_window(
    const torch::Tensor& r, double cutoff, double min_cutoff, double switch_width) {
  auto weight = torch::ones_like(r);
  const double width = resolve_dispersion_switch_width(cutoff, min_cutoff, switch_width);
  if (cutoff > 0.0) {
    if (width > 0.0) {
      weight = weight * quintic_smoothstep((cutoff - r) / width);
    } else {
      weight = weight * (r <= cutoff).to(r.scalar_type());
    }
  }
  if (min_cutoff > 0.0) {
    if (width > 0.0) {
      weight = weight * quintic_smoothstep((r - min_cutoff) / width);
    } else {
      weight = weight * (r >= min_cutoff).to(r.scalar_type());
    }
  }
  return weight;
}

// per-axis assignment weights w[0..S-1] (each [N]) for the in-cell offset f in [0,1).
// Mirrors Python _assignment_kernel_1d (cic: linear; pcs: cubic B-spline).
static std::vector<torch::Tensor> axis_weights(const torch::Tensor& f, int a) {
  if (a == 1) {  // pcs (cubic B-spline) -> C^2 weight, C^1 force, no mesh-cell discontinuity
    auto f2 = f * f, f3 = f2 * f;
    return {torch::pow(1.0 - f, 3) / 6.0,
            (3.0 * f3 - 6.0 * f2 + 4.0) / 6.0,
            (-3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0) / 6.0,
            f3 / 6.0};
  }
  return {1.0 - f, f};  // cic (linear)
}

torch::Tensor MFFMBDSolver::k_grid_cart(const torch::Tensor& eff_cell, const torch::Device& device) const {
  const int M = config_.mesh_size;
  auto opt = torch::TensorOptions().dtype(eff_cell.scalar_type()).device(device);  // on-device, input dtype
  auto freq = fft_freqs(M, opt);
  auto grids = torch::meshgrid({freq, freq, freq}, "ij");
  auto ik = torch::stack({grids[0], grids[1], grids[2]}, -1).reshape({-1, 3});  // [K,3]
  auto inv = torch::linalg_inv(eff_cell.to(device));
  return 2.0 * M_PI * torch::matmul(ik, inv.transpose(0, 1));                 // [K,3]
}

// flat mesh indices [N,S^3] + stencil weights [N,S^3] in one shot (mirrors Python; one outer product +
// one cartesian-prod offset add -> no S^3 launch loop). weights inherit the frac dtype (f32 or f64).
void MFFMBDSolver::assignment_idx_weights(const torch::Tensor& frac, torch::Tensor& flat_idx,
                                          torch::Tensor& weights) const {
  const int M = config_.mesh_size, a = config_.assignment, S = assignment_stencil(a);
  const int64_t N = frac.size(0);
  auto scaled = frac * (double)M;
  auto floor_scaled = torch::floor(scaled);
  auto f = scaled - floor_scaled;                                          // [N,3] in [0,1)
  auto base = floor_scaled.to(torch::kLong) - (a == 1 ? 1 : 0);           // pcs base = floor-1
  auto WX = axis_weights(f.select(1, 0), a), WY = axis_weights(f.select(1, 1), a),
       WZ = axis_weights(f.select(1, 2), a);
  auto wx = torch::stack(WX, 1), wy = torch::stack(WY, 1), wz = torch::stack(WZ, 1);  // [N,S]
  weights = (wx.view({N, S, 1, 1}) * wy.view({N, 1, S, 1}) * wz.view({N, 1, 1, S})).reshape({N, S * S * S});
  auto rng = torch::arange(S, torch::TensorOptions().dtype(torch::kLong).device(frac.device()));
  auto offs = torch::cartesian_prod({rng, rng, rng});                      // [S^3,3], (ix slow .. iz fast)
  auto idx = torch::remainder(base.unsqueeze(1) + offs.unsqueeze(0), M);   // [N,S^3,3]
  flat_idx = (idx.select(2, 0) * M + idx.select(2, 1)) * M + idx.select(2, 2);  // [N,S^3]
}

torch::Tensor MFFMBDSolver::spread_to_mesh(const torch::Tensor& frac, const torch::Tensor& source,
                                           const std::array<int, 3>&) const {
  const int M = config_.mesh_size, C = source.size(1);
  torch::Tensor flat_idx, weights;
  assignment_idx_weights(frac, flat_idx, weights);                        // [N,S^3],[N,S^3]
  auto mesh = torch::zeros({M * M * M, C}, source.options());
  auto vals = (source.unsqueeze(1) * weights.unsqueeze(-1)).reshape({-1, C});  // [N*S^3, C]
  mesh.scatter_add_(0, flat_idx.reshape({-1, 1}).expand({-1, C}), vals);  // ONE scatter_add
  return mesh.view({M, M, M, C});
}

torch::Tensor MFFMBDSolver::gather_from_mesh(const torch::Tensor& frac, const torch::Tensor& mesh,
                                             const std::array<int, 3>&) const {
  const int M = config_.mesh_size, C = mesh.size(-1), S = assignment_stencil(config_.assignment);
  const int64_t N = frac.size(0);
  torch::Tensor flat_idx, weights;
  assignment_idx_weights(frac, flat_idx, weights);
  auto flat_mesh = mesh.view({-1, C});
  auto gathered = flat_mesh.index_select(0, flat_idx.reshape(-1)).view({N, S * S * S, C});  // ONE gather
  return (gathered * weights.unsqueeze(-1)).sum(1);                       // [N,C]
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> MFFMBDSolver::build_pme_fft_cache(
    const torch::Tensor& pos, const torch::Tensor& cell, double alpha, const torch::Device& device) const {
  const int64_t N = pos.size(0);
  const int M = config_.mesh_size;
  auto eff = cell.to(device);
  auto inv = torch::linalg_inv(eff);
  auto frac = torch::matmul(pos, inv);
  frac = frac - torch::floor(frac);
  auto k_cart = k_grid_cart(eff, device);                                      // [K,3]
  auto k2 = (k_cart * k_cart).sum(-1).clamp_min(1e-12);                         // [K]
  auto volume = torch::det(eff).abs().clamp_min(1e-6);
  auto screen = torch::exp(-k2 / (4.0 * alpha * alpha));
  auto dopt = torch::TensorOptions().dtype(eff.scalar_type()).device(device);
  auto sinc1d = torch::sinc(fft_freqs(M, dopt) / (double)M).pow(assignment_stencil(config_.assignment));
  auto win = (sinc1d.view({M, 1, 1}) * sinc1d.view({1, M, 1}) * sinc1d.view({1, 1, M})).reshape({-1});
  auto wdeconv = torch::reciprocal(win.clamp_min(1e-6).square());               // [K]
  auto scale = -(4.0 * M_PI) / volume * screen * wdeconv / k2;                  // includes /k^2
  scale = torch::where(k2 > 1e-12, scale, torch::zeros_like(scale));            // k=0 tinfoil

  auto kc = k_cart.to(k_cart.scalar_type() == torch::kFloat64 ? torch::kComplexDouble : torch::kComplexFloat);
  auto kernel_fft = scale.to(kc.dtype()).unsqueeze(-1).unsqueeze(-1)
                  * kc.unsqueeze(-1) * kc.unsqueeze(-2);                       // [K,3,3]
  auto kernel_mesh = torch::real(torch::fft::ifftn(kernel_fft.reshape({M, M, M, 9}), {}, {0, 1, 2}))
                     .reshape({M * M * M, 3, 3}) * std::pow((double)M, 3);
  torch::Tensor self_idx, self_w;
  assignment_idx_weights(frac, self_idx, self_w);                              // [N,S3], [N,S3]
  const int64_t S3 = self_idx.size(1);
  auto ix = torch::floor_divide(self_idx, M * M);
  auto iy = torch::remainder(torch::floor_divide(self_idx, M), M);
  auto iz = torch::remainder(self_idx, M);
  auto self_blocks = torch::zeros({N, 3, 3}, eff.options());
  const int64_t max_pairs_per_chunk = 262144;
  const int64_t chunk = std::max<int64_t>(1, std::min<int64_t>(N, max_pairs_per_chunk / std::max<int64_t>(S3 * S3, 1)));
  for (int64_t start = 0; start < N; start += chunk) {
    const int64_t len = std::min<int64_t>(chunk, N - start);
    auto ix_c = ix.narrow(0, start, len);
    auto iy_c = iy.narrow(0, start, len);
    auto iz_c = iz.narrow(0, start, len);
    auto dx = torch::remainder(ix_c.unsqueeze(2) - ix_c.unsqueeze(1), M);
    auto dy = torch::remainder(iy_c.unsqueeze(2) - iy_c.unsqueeze(1), M);
    auto dz = torch::remainder(iz_c.unsqueeze(2) - iz_c.unsqueeze(1), M);
    auto flat = (dx * M + dy) * M + dz;
    auto vals = kernel_mesh.index_select(0, flat.reshape(-1)).reshape({len, S3, S3, 3, 3});
    auto w = self_w.narrow(0, start, len);
    auto ww = w.unsqueeze(2) * w.unsqueeze(1);
    auto block = (vals * ww.unsqueeze(-1).unsqueeze(-1)).sum(std::vector<int64_t>{1, 2});
    self_blocks.narrow(0, start, len).copy_(block);
  }
  return std::make_tuple(frac, k_cart, scale, self_blocks);
}

torch::Tensor MFFMBDSolver::dipole_field(
    const torch::Tensor& pos, const torch::Tensor& mu, const torch::Tensor& cell, double alpha,
    const torch::Tensor& src, const torch::Tensor& dst, const torch::Tensor& shifts,
    const torch::Device& device, const torch::Tensor& alpha_pol, double beta, bool damping,
    const torch::Tensor& pme_frac, const torch::Tensor& pme_k_cart, const torch::Tensor& pme_scale,
    const torch::Tensor& pme_self_blocks) const {
  const int64_t N = pos.size(0);
  auto eff = cell.to(device);  // inherit input dtype (float32 on GPU; float64 in the tests)
  auto field = torch::zeros_like(mu);
  // mu is [N,3] (one vector) or [P,N,3] (P Hutchinson probes batched) -> ONE matvec over all probes.
  const bool batched = mu.dim() == 3;
  const int64_t P = batched ? mu.size(0) : 1;

  if (config_.use_fft) {
    // ---- pme_fft: mirror the trained apply_periodic_dipole_pme_field EXACTLY -- a reciprocal-ONLY smooth
    // screened mesh operator (NO erfc near field, exact discrete self block removed, NO edge list).
    // The P probes ride as FFT channels.
    // MBD uses T=grad grad(1/r)=(3rr-I)/r^3, so the reciprocal tensor is
    // spectral = -(4pi/V) wdeconv e^{-k^2/4a^2}; field_fft = spectral k (k.mu)/k^2.
    // alpha = prefactor/(0.5 Lmin).
    const int M = config_.mesh_size;
    torch::Tensor frac, k_cart, scale, self_blocks;
    if (pme_frac.defined() && pme_k_cart.defined() && pme_scale.defined() && pme_self_blocks.defined()) {
      frac = pme_frac;
      k_cart = pme_k_cart;
      scale = pme_scale;
      self_blocks = pme_self_blocks;
    } else {
      std::tie(frac, k_cart, scale, self_blocks) = build_pme_fft_cache(pos, cell, alpha, device);
    }
    // pack the probe batch into the spread channel: [P,N,3] -> [N, P*3]  (no-op for [N,3]).
    auto mu_nc = batched ? mu.permute({1, 0, 2}).reshape({N, P * 3}).contiguous() : mu;
    auto mu_mesh = spread_to_mesh(frac, mu_nc, config_.pbc);                  // [M,M,M, P*3]
    auto mu_k = torch::fft::fftn(mu_mesh, {}, {0, 1, 2}).reshape({-1, P, 3}); // [K,P,3] complex
    auto kc = k_cart.to(mu_k.dtype());                                        // [K,3]
    auto kdotmu = (kc.unsqueeze(1) * mu_k).sum(-1);                           // [K,P] complex
    auto e_k = (scale.to(mu_k.dtype()).unsqueeze(-1).unsqueeze(-1) * kc.unsqueeze(1)) * kdotmu.unsqueeze(-1);  // [K,P,3]
    auto e_mesh = torch::real(torch::fft::ifftn(e_k.reshape({M, M, M, P * 3}), {}, {0, 1, 2})) * std::pow((double)M, 3);
    auto gathered = gather_from_mesh(frac, e_mesh, config_.pbc);              // [N, P*3]
    field = batched ? gathered.reshape({N, P, 3}).permute({1, 0, 2}).contiguous() : gathered;
    if (batched) {
      auto self_field = torch::matmul(self_blocks.unsqueeze(1), mu.permute({1, 0, 2}).unsqueeze(-1))
                            .squeeze(-1).permute({1, 0, 2}).contiguous();
      field = field - self_field;
    } else {
      field = field - torch::matmul(self_blocks, mu.unsqueeze(-1)).squeeze(-1);
    }
  } else if (src.numel() > 0) {
    // ---- edge_sparse (default): field += sum_edges damp*T_bare.mu  (matches the trained edge_sparse
    // operator; no FFT/reciprocal/self -> the diagonal omega^2 lives in coupled_matvec). O(E) -> fast. ----
    const int64_t bd = mu.dim() - 2;  // the N axis: 0 for [N,3], 1 for [P,N,3]
    auto shift_cart = torch::matmul(shifts.to(eff.scalar_type()), eff);
    auto rvec = pos.index_select(0, dst) - pos.index_select(0, src) + shift_cart;  // [E,3]
    auto r = torch::linalg_vector_norm(rvec, 2, -1).clamp_min(1e-12);
    auto r2 = r * r, r3 = r2 * r, r5 = r3 * r2;                               // [E]
    auto mu_src = mu.index_select(bd, src);                                   // [...,E,3]
    auto rdotmu = (rvec * mu_src).sum(-1);                                    // [...,E]  (rvec [E,3] broadcasts)
    // bare dipole tensor contraction: T_bare.mu = 3(r.mu)r/r^5 - mu/r^3,  T_bare = (3 rr - I)/r^3.
    auto t_bare = 3.0 * rdotmu.unsqueeze(-1) * rvec / r5.unsqueeze(-1) - mu_src / r3.unsqueeze(-1);  // [...,E,3]
    // rsSCS damping: damp = 1 - exp(-(r/(beta*R))^6), R = a_i^1/3 + a_j^1/3.
    torch::Tensor damp;
    if (damping && alpha_pol.defined()) {
      auto radius = (alpha_pol.index_select(0, src).clamp_min(0.0).pow(1.0 / 3.0)
                     + alpha_pol.index_select(0, dst).clamp_min(0.0).pow(1.0 / 3.0)).clamp_min(1e-6);
      damp = 1.0 - torch::exp(-torch::pow((r / (beta * radius)).clamp_min(0.0), 6));
    } else {
      damp = torch::ones_like(r);
    }
    auto window = dispersion_edge_window(r, config_.real_cutoff, config_.min_cutoff, config_.switch_width);
    field = field.index_add(bd, dst, (damp * window).unsqueeze(-1) * t_bare); // [...,E,3] -> [...,N,3]
  }
  return field;
}

torch::Tensor MFFMBDSolver::coupled_matvec(
    const torch::Tensor& x, const torch::Tensor& omega, const torch::Tensor& alpha,
    const torch::Tensor& pos, const torch::Tensor& cell, double alpha_ewald,
    const torch::Tensor& src, const torch::Tensor& dst, const torch::Tensor& shifts,
    const torch::Device& device, const torch::Tensor& W,
    const torch::Tensor& pme_frac, const torch::Tensor& pme_k_cart, const torch::Tensor& pme_scale,
    const torch::Tensor& pme_self_blocks) const {
  // Off-diagonal C_ij = coupling_scale * damp(r) * W_i T_ij W_j  (W = omega*alpha^{1/2}). ANISOTROPIC: W is
  // a per-atom 3x3 (mu_i = W_i x_i, matmul). ISOTROPIC: W undefined -> scalar wsa = omega*sqrt(alpha).
  const bool aniso = W.defined();
  auto wsa = aniso ? torch::Tensor() : (omega * alpha.clamp_min(0).sqrt()).unsqueeze(-1);   // [N,1]
  auto mu = aniso ? torch::matmul(W, x.unsqueeze(-1)).squeeze(-1) : (wsa * x);
  auto fld = dipole_field(pos, mu, cell, alpha_ewald, src, dst, shifts, device,
                          alpha, config_.mbd_beta, config_.damping,
                          pme_frac, pme_k_cart, pme_scale, pme_self_blocks);
  auto coupled = aniso ? torch::matmul(W, fld.unsqueeze(-1)).squeeze(-1) : (wsa * fld);
  return (omega.unsqueeze(-1) * omega.unsqueeze(-1)) * x + config_.coupling_scale * coupled;
}

// Reconstruct the per-atom coupling factor W = omega * B from an [N,8] anisotropic source
// [omega, alpha_iso, Bxx,Byy,Bzz,Bxy,Bxz,Byz]; returns an undefined tensor for an isotropic [N,2] source.
torch::Tensor MFFMBDSolver::build_W(const torch::Tensor& src, const torch::Tensor& omega) const {
  if (src.size(1) < 8) return {};
  auto Bxx = src.select(1, 2), Byy = src.select(1, 3), Bzz = src.select(1, 4);
  auto Bxy = src.select(1, 5), Bxz = src.select(1, 6), Byz = src.select(1, 7);
  auto r0 = torch::stack({Bxx, Bxy, Bxz}, 1);
  auto r1 = torch::stack({Bxy, Byy, Byz}, 1);
  auto r2 = torch::stack({Bxz, Byz, Bzz}, 1);
  return omega.view({-1, 1, 1}) * torch::stack({r0, r1, r2}, 1);   // [N,3,3]
}

double MFFMBDSolver::mbd_energy_dense(
    const torch::Tensor& global_pos, const torch::Tensor& mbd_source, const torch::Tensor& cell,
    const torch::Tensor& src, const torch::Tensor& dst, const torch::Tensor& shifts,
    const torch::Device& device, double alpha_ewald, double* min_eig) const {
  torch::NoGradGuard ng;
  const int N = global_pos.size(0), n = 3 * N;
  auto omega = mbd_source.select(1, 0).contiguous();
  auto alpha = mbd_source.select(1, 1).contiguous();
  auto W = build_W(mbd_source, omega);                // anisotropic [N,3,3] (undefined for isotropic [N,2])
  double Lmin = torch::linalg_vector_norm(cell.to(torch::kFloat64), 2, 1).min().item<double>();
  double alpha_ew = (alpha_ewald > 0.0) ? alpha_ewald : config_.ewald_alpha_prefactor / (0.5 * Lmin);
  auto opt = torch::TensorOptions().dtype(torch::kFloat64).device(device);
  std::vector<torch::Tensor> cols;
  for (int i = 0; i < n; ++i) {
    auto e = torch::zeros({n}, opt); e[i] = 1.0;
    cols.push_back(coupled_matvec(e.view({N, 3}), omega, alpha, global_pos, cell, alpha_ew, src, dst, shifts, device, W)
                       .reshape({-1}));
  }
  auto C = torch::stack(cols, 1);                  // [n,n], column i = C.e_i
  C = 0.5 * (C + C.transpose(0, 1));               // symmetrize numerical asymmetry
  auto eig_raw = torch::linalg_eigvalsh(C);
  if (min_eig) *min_eig = eig_raw.min().item<double>();   // <0 -> C indefinite (QHO unphysical)
  auto eig = eig_raw.clamp_min(1e-10);
  return config_.energy_scale * (0.5 * eig.sqrt().sum() - 1.5 * omega.sum()).item<double>();
}

torch::Tensor MFFMBDSolver::mbd_energy(
    const torch::Tensor& global_pos, const torch::Tensor& mbd_source, const torch::Tensor& cell,
    const torch::Tensor& src, const torch::Tensor& dst, const torch::Tensor& shifts,
    const torch::Device& device, double alpha_ewald, double* used_lmin, double* used_lmax) const {
  const int N = global_pos.size(0);
  const int n = 3 * N;
  auto omega = mbd_source.select(1, 0).contiguous();   // [N]
  auto alpha = mbd_source.select(1, 1).contiguous();   // [N]  (isotropic alpha, or alpha_iso for the damping)
  auto W = build_W(mbd_source, omega);                 // anisotropic [N,3,3] (undefined for isotropic [N,2])
  // alpha (Ewald) = prefactor / (0.5 * min periodic box length)
  auto rownorm = torch::linalg_vector_norm(cell.to(torch::kFloat64), 2, 1);
  double Lmin = rownorm.min().item<double>();
  double alpha_ew = (alpha_ewald > 0.0) ? alpha_ewald : config_.ewald_alpha_prefactor / (0.5 * Lmin);
  torch::Tensor pme_frac, pme_k_cart, pme_scale, pme_self_blocks;
  if (config_.use_fft) {
    std::tie(pme_frac, pme_k_cart, pme_scale, pme_self_blocks) =
        build_pme_fft_cache(global_pos, cell, alpha_ew, device);
  }

  auto mv = [&](const torch::Tensor& v) {
    // v is [n] (one vector) or [P,n] (P probes) -> [.,N,3] -> coupled matvec -> back to v's shape.
    auto y = coupled_matvec(
        v.reshape({-1, N, 3}), omega, alpha, global_pos, cell, alpha_ew, src, dst, shifts, device, W,
        pme_frac, pme_k_cart, pme_scale, pme_self_blocks);
    return y.reshape(v.sizes());
  };

  // --- spectral bounds: fixed (config) for conservative MD, else matvec-only power iteration ---
  auto opt = torch::TensorOptions().dtype(global_pos.scalar_type()).device(device);  // inherit f32/f64
  double lmax, lmin;
  if (config_.cheb_lmax > 0.0) {
    lmin = config_.cheb_lmin; lmax = config_.cheb_lmax;  // E becomes a smooth fn of p -> conservative
  } else {
    torch::NoGradGuard ng;
    // deterministic inits (CPU generator -> device) so bounds, hence E, are reproducible across calls.
    auto g1 = at::detail::createCPUGenerator(12345);
    auto v = torch::randn({n}, g1, torch::TensorOptions().dtype(torch::kFloat64)).to(opt);
    v = v / v.norm();
    double lam = 0;
    for (int i = 0; i < config_.power_steps; ++i) { auto w = mv(v); lam = (v * w).sum().item<double>(); v = w / w.norm().clamp_min(1e-30); }
    lmax = lam * (1.0 + config_.bound_pad);
    auto g2 = at::detail::createCPUGenerator(67890);
    v = torch::randn({n}, g2, torch::TensorOptions().dtype(torch::kFloat64)).to(opt); v = v / v.norm();
    double mu = 0;
    for (int i = 0; i < config_.power_steps; ++i) { auto w = lmax * v - mv(v); mu = (v * w).sum().item<double>(); v = w / w.norm().clamp_min(1e-30); }
    lmin = std::max((lmax - mu) * (1.0 - config_.bound_pad), 1e-6);
  }
  if (used_lmin) *used_lmin = lmin;
  if (used_lmax) *used_lmax = lmax;

  // --- Chebyshev coeffs of sqrt on [lmin,lmax] ---
  const int deg = config_.cheb_degree, Mc = deg + 1;
  auto j = torch::arange(Mc, opt) + 0.5;
  auto theta = M_PI * j / (double)Mc;
  auto xx = 0.5 * (lmax + lmin) + 0.5 * (lmax - lmin) * torch::cos(theta);
  auto fx = xx.clamp_min(0).sqrt();
  auto kk = torch::arange(deg + 1, opt);
  auto coef = (2.0 / Mc) * (fx.unsqueeze(0) * torch::cos(kk.unsqueeze(1) * theta.unsqueeze(0))).sum(1);
  coef.index_put_({0}, coef.index({0}) * 0.5);

  double a = 2.0 / (lmax - lmin), b = -(lmax + lmin) / (lmax - lmin);
  auto smv = [&](const torch::Tensor& v) { return a * mv(v) + b * v; };

  // --- Hutchinson + Chebyshev recurrence (fixed Rademacher probes) ---
  auto gen = at::detail::createCPUGenerator(0);
  auto Z = (2.0 * torch::randint(0, 2, {config_.num_probes, n}, gen, torch::TensorOptions().dtype(torch::kFloat64)) - 1.0).to(opt);  // [P,n]
  // BATCHED Hutchinson + Chebyshev: all P probes ride ONE recurrence -> degree (not P*degree) sequential
  // matvecs. Numerically identical to the per-probe loop (same fixed probes), just one batched matvec/step.
  auto t_prev = Z;
  auto t_cur = smv(Z);                                                  // [P,n]
  auto s = coef[0] * (Z * Z).sum(-1) + coef[1] * (Z * t_cur).sum(-1);   // [P]
  for (int kx = 2; kx <= deg; ++kx) {
    auto t_next = 2.0 * smv(t_cur) - t_prev;                            // [P,n]
    s = s + coef[kx] * (Z * t_next).sum(-1);                            // [P]
    t_prev = t_cur; t_cur = t_next;
  }
  auto tr = s.mean();                                                   // (1/P) sum_p z_p^T sqrt(C) z_p
  return config_.energy_scale * (0.5 * tr - 1.5 * omega.sum());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MFFMBDSolver::build_periodic_neighbors(
    const torch::Tensor& pos, const torch::Tensor& cell, double cutoff) const {
  auto opt = torch::TensorOptions().dtype(torch::kFloat64).device(pos.device());
  auto cellf = cell.to(torch::kFloat64);
  auto rownorm = torch::linalg_vector_norm(cellf, 2, 1);  // [3]
  int na = config_.pbc[0] ? (int)std::ceil(cutoff / rownorm[0].item<double>()) : 0;
  int nb = config_.pbc[1] ? (int)std::ceil(cutoff / rownorm[1].item<double>()) : 0;
  int nc = config_.pbc[2] ? (int)std::ceil(cutoff / rownorm[2].item<double>()) : 0;
  std::vector<torch::Tensor> vsrc, vdst, vshift;
  auto pi = pos.unsqueeze(1);  // [N,1,3]
  auto pj = pos.unsqueeze(0);  // [1,N,3]
  for (int i = -na; i <= na; ++i)
    for (int j = -nb; j <= nb; ++j)
      for (int k = -nc; k <= nc; ++k) {
        auto shift_int = torch::tensor({(double)i, (double)j, (double)k}, opt);     // [3]
        auto shift_cart = torch::matmul(shift_int, cellf);                          // [3]
        auto rij = pj + shift_cart - pi;                                            // [N,N,3]  pos[q]+s-pos[p]
        auto dist = torch::linalg_vector_norm(rij, 2, -1);                          // [N,N]
        auto within = dist < cutoff;
        if (i == 0 && j == 0 && k == 0) within = within & (dist > 1e-8);            // drop self
        auto idx = torch::nonzero(within);                                          // [E,2] (p,q)
        if (idx.size(0) == 0) continue;
        vsrc.push_back(idx.select(1, 0));
        vdst.push_back(idx.select(1, 1));
        vshift.push_back(shift_int.unsqueeze(0).expand({idx.size(0), 3}).contiguous());
      }
  if (vsrc.empty()) {
    auto e = torch::empty({0}, torch::TensorOptions().dtype(torch::kLong).device(pos.device()));
    return {e, e, torch::empty({0, 3}, opt)};
  }
  return {torch::cat(vsrc), torch::cat(vdst), torch::cat(vshift)};
}

MBDOutputs MFFMBDSolver::run_autograd(const torch::Tensor& pos, const torch::Tensor& source,
                                      const torch::Tensor& cell, const torch::Tensor& src,
                                      const torch::Tensor& dst, const torch::Tensor& shift,
                                      double alpha_ewald, const torch::Device& device) const {
  const int N = pos.size(0);
  torch::AutoGradMode grad_on(true);
  auto p = pos.to(device).clone().detach().set_requires_grad(true);  // inherit input dtype (f32 on GPU)
  auto cellf = cell.to(device);
  auto src_ = source.to(device);
  double bl = 0.0, bu = 0.0;
  auto E = mbd_energy(p, src_, cellf, src, dst, shift, device, alpha_ewald, &bl, &bu);
  OptionalGilRelease no_gil;
  auto grads = torch::autograd::grad({E}, {p}, /*grad_outputs=*/{}, /*retain_graph=*/false,
                                     /*create_graph=*/false, /*allow_unused=*/true);
  MBDOutputs out;
  out.lmin = bl; out.lmax = bu;
  out.energy = E.item<double>();
  out.forces = grads[0].defined() ? (-grads[0]).detach()
                                  : torch::zeros({N, 3}, torch::TensorOptions().dtype(p.scalar_type()).device(device));
  out.atom_energy = torch::full({N}, out.energy / N, torch::TensorOptions().dtype(p.scalar_type()).device(device));
  return out;
}

int MFFMBDSolver::adaptive_mesh(double alpha, const torch::Tensor& cell) const {
  // FFT Nyquist k_Nyq = pi*M/L must resolve the e^{-k^2/4a^2} screen -> M ~ mesh_per_alpha * alpha * L.
  double Lmax = torch::linalg_vector_norm(cell.to(torch::kFloat64), 2, 1).max().item<double>();
  int target = (int)std::ceil(config_.mesh_per_alpha * alpha * Lmax);
  // Round to a POWER-OF-2 mesh: non-power-of-2 sizes (e.g. 38=2*19) hit slow cuFFT (Bluestein) paths;
  // a power of 2 is the fast radix-2 path. Clamp to [mesh_min, mesh_max].
  int M = config_.mesh_min;
  while (M < target && M < config_.mesh_max) M *= 2;
  if (M > config_.mesh_max) M = config_.mesh_max;
  return M;
}

// Deployment path: reuse a LAMMPS real-space list at `real_cutoff`; alpha tied to the cutoff, mesh
// adapted to alpha (the reciprocal far field has no cutoff -> "dispersion range > r_cut" is fine).
MBDOutputs MFFMBDSolver::compute(const torch::Tensor& pos, const torch::Tensor& source,
                                 const torch::Tensor& cell, const torch::Tensor& src,
                                 const torch::Tensor& dst, const torch::Tensor& shifts,
                                 double real_cutoff, const torch::Device& device) {
  // alpha_ewald=-1 -> mbd_energy derives the BOX-tied alpha = ewald_alpha_prefactor/(0.5 Lmin). For pme_fft
  // (use_fft) that matches the trained build_periodic_dipole_pme_kernel exactly, with the FIXED training mesh
  // (config_.mesh_size). For edge_sparse (direct) dipole_field ignores alpha + mesh entirely.
  // The edge list is still supplied by LAMMPS, but the C2 cutoff window must use the same trained/deployed
  // dispersion radius as Python. Keep it on the solver config so every Chebyshev matvec sees the same window.
  if (real_cutoff > 0.0) config_.real_cutoff = real_cutoff;
  return run_autograd(pos, source, cell, src, dst, shifts, /*alpha_ewald=*/-1.0, device);
}

// Fallback path (no LAMMPS edge list): alpha tied to the box. pme_fft (use_fft) is reciprocal-only -> pos+
// cell + the FIXED training mesh, NO neighbor list. edge_sparse (direct) builds an O(N^2) periodic list.
MBDOutputs MFFMBDSolver::compute(const torch::Tensor& pos, const torch::Tensor& source,
                                 const torch::Tensor& cell, const torch::Device& device) {
  auto cellf = cell.to(torch::kFloat64);
  double Lmin = torch::linalg_vector_norm(cellf, 2, 1).min().item<double>();
  double alpha_ew = config_.ewald_alpha_prefactor / (0.5 * Lmin);  // box-tied; pme_fft matches training
  torch::Tensor src, dst, shift;
  if (config_.use_fft) {
    // pme_fft: dipole_field uses pos+cell on the FIXED config_.mesh_size (the trained mesh); no edges.
    auto lopt = torch::TensorOptions().dtype(torch::kLong).device(pos.device());
    src = torch::empty({0}, lopt);
    dst = torch::empty({0}, lopt);
    shift = torch::zeros({0, 3}, cellf.options().device(pos.device()));
  } else {
    double cutoff = config_.real_cutoff > 0.0 ? config_.real_cutoff : 5.0 / alpha_ew;  // erfc(5)~1e-12
    torch::NoGradGuard ng;
    std::tie(src, dst, shift) = build_periodic_neighbors(pos.to(device, torch::kFloat64), cellf, cutoff);
  }
  return run_autograd(pos, source, cell, src, dst, shift, alpha_ew, device);
}

}  // namespace mfftorch

#ifdef MBD_STANDALONE_TEST
#include <cstdio>
// Deterministic parity harness: a HARDCODED tiny system (identical in the Python reference,
// /tmp/mbd_parity_ref.py) -> compares the dipole_field operator (reciprocal PME + T_SR + self).
// No random / no cross-language tensor passing: prints two invariants of the field.
int main() {
  using namespace mfftorch;
  auto o = torch::TensorOptions().dtype(torch::kFloat64);
  auto pos = torch::tensor({{1.0, 1.0, 1.0}, {3.0, 1.5, 1.2}}, o);
  auto mu = torch::tensor({{0.3, -0.5, 0.8}, {0.1, 0.2, -0.4}}, o);
  auto cell = torch::eye(3, o) * 10.0;
  auto src = torch::tensor({0, 1}, torch::TensorOptions().dtype(torch::kLong));
  auto dst = torch::tensor({1, 0}, torch::TensorOptions().dtype(torch::kLong));
  auto sh = torch::zeros({2, 3}, o);
  const char* names[2] = {"CIC", "PCS"};
  for (int a = 0; a < 2; ++a) {  // verify BOTH assignments match the Python backend
    MBDConfig cfg; cfg.mesh_size = 32; cfg.assignment = a; cfg.use_fft = true;  // parity vs Python PME
    MFFMBDSolver solver; solver.set_config(cfg);
    auto f = solver.dipole_field(pos, mu, cell, /*alpha=*/1.0, src, dst, sh, torch::kCPU);
    std::printf("CPP_%s_SUM %.10f\n", names[a], f.sum().item<double>());
    std::printf("CPP_%s_SQ %.10f\n", names[a], (f * f).sum().item<double>());
  }
  // full solver end-to-end (pcs): source [N,2] = (omega, alpha); power-iter bounds + Chebyshev energy.
  MBDConfig cfg; cfg.mesh_size = 32; cfg.cheb_degree = 20; cfg.num_probes = 64; cfg.use_fft = true;  // pcs
  MFFMBDSolver solver; solver.set_config(cfg);
  auto source = torch::tensor({{1.0, 0.3}, {1.1, 0.3}}, o);  // (omega, alpha_pol)
  auto E = solver.mbd_energy(pos, source, cell, src, dst, sh, torch::kCPU);
  std::printf("CPP_E_MBD %.8f (finite=%d)\n", E.item<double>(), (int)std::isfinite(E.item<double>()));
  return 0;
}
#endif

#ifdef MBD_MD_TEST
#include <cstdio>
#include <vector>
// Demonstrates PCS (4th-order) gives CONSERVATIVE forces at mesh-cell boundaries where 2nd-order CIC
// does NOT. The crystal sits ON the grid (z = k*3 -> scaled 8k = integer at mesh 24; atoms 0=(0,0,0)
// and 26=(6,6,6) fully on boundaries), so the boundary-aligned components are exactly the CIC
// force-discontinuity case. Per assignment: fix spectral bounds (-> deterministic surrogate) then
// check autograd force == central finite difference.
int main() {
  using namespace mfftorch;
  auto o = torch::TensorOptions().dtype(torch::kFloat64);
  const int nx = 3; const double a = 3.0; const double L = nx * a;
  std::vector<std::array<double, 3>> P;
  for (int i = 0; i < nx; ++i) for (int j = 0; j < nx; ++j) for (int k = 0; k < nx; ++k)
    P.push_back({i * a + 0.15 * ((i + j + k) % 2), j * a + 0.1 * (k % 2), k * a});  // z = k*3, ON the grid
  const int N = (int)P.size();
  auto pos = torch::zeros({N, 3}, o);
  for (int n = 0; n < N; ++n) { pos[n][0] = P[n][0]; pos[n][1] = P[n][1]; pos[n][2] = P[n][2]; }
  auto cell = torch::eye(3, o) * L;
  auto source = torch::zeros({N, 2}, o);
  for (int n = 0; n < N; ++n) { source[n][0] = (n % 2 ? 1.1 : 1.0); source[n][1] = 0.30; }

  const char* names[2] = {"CIC", "PCS"};
  for (int asg = 0; asg < 2; ++asg) {
    // mesh 12: z=k*3 -> scaled 4k = integer -> atoms 0,26 ON cell boundaries; small for the FD sweep.
    MBDConfig cfg; cfg.assignment = asg; cfg.mesh_size = cfg.mesh_min = cfg.mesh_max = 12; cfg.use_fft = true;
    cfg.cheb_degree = 14; cfg.num_probes = 16; cfg.ewald_alpha_prefactor = 5.0;  // mesh-boundary conservativity
    MFFMBDSolver solver; solver.set_config(cfg);
    auto out = solver.compute(pos, source, cell, torch::kCPU);
    cfg.cheb_lmin = out.lmin / 1.2; cfg.cheb_lmax = out.lmax * 1.2; solver.set_config(cfg);
    out = solver.compute(pos, source, cell, torch::kCPU);
    const double h = 1e-5; double worst = 0.0;
    auto Eat = [&](int atom, int comp, double dh) { auto pp = pos.clone(); pp[atom][comp] += dh;
                                                    return solver.compute(pp, source, cell, torch::kCPU).energy; };
    for (int atom : {0, N / 2, N - 1})
      for (int comp = 0; comp < 3; ++comp) {
        double fd = -(Eat(atom, comp, h) - Eat(atom, comp, -h)) / (2 * h);
        double fa = out.forces[atom][comp].item<double>();
        worst = std::max(worst, std::abs(fa - fd) / (std::abs(fd) + 1e-6));
      }
    std::printf("assignment=%s mesh=%d (atoms ON cell boundaries): E=%.8f worst conservativity rel err = %.2e  %s\n",
                names[asg], cfg.mesh_min, out.energy, worst, worst < 1e-3 ? "CONSERVATIVE" : "DISCONTINUOUS-FORCE");
  }
  return 0;
}
#endif

#ifdef MBD_PHYS_TEST
#include <cstdio>
// Validates the rsSCS damping fixes the E_MBD physicality (the "sign"). A CLOSE pair (r=1.2, inside the
// damping radius): the UNDAMPED dipole tensor over-couples and makes the coupled-dipole matrix C
// INDEFINITE (min eigenvalue < 0 -> imaginary QHO frequency, unphysical, the E_MBD pathology). The
// rsSCS damping keeps C positive-definite (min eigenvalue > 0) -> physical, E_MBD <= 0. Compared to the
// Python dense reference (/tmp/mbd_phys_ref.py).
int main() {
  using namespace mfftorch;
  auto o = torch::TensorOptions().dtype(torch::kFloat64);
  auto lo = torch::TensorOptions().dtype(torch::kLong);
  auto pos = torch::tensor({{0.0, 0.0, 0.0}, {1.2, 0.0, 0.0}}, o);
  const double L = 40.0;
  auto cell = torch::eye(3, o) * L;
  auto source = torch::tensor({{1.0, 1.0}, {1.0, 1.0}}, o);  // (omega, alpha) -> radius = 1+1 = 2
  auto src = torch::tensor({1, 0}, lo);
  auto dst = torch::tensor({0, 1}, lo);
  auto sh = torch::zeros({2, 3}, o);
  MBDConfig cfg; cfg.mesh_size = 48; cfg.coupling_scale = 1.0; cfg.mbd_beta = 1.5;
  MFFMBDSolver solver;
  double me_u = 0.0, me_d = 0.0;
  cfg.damping = false; solver.set_config(cfg);
  double Eu = solver.mbd_energy_dense(pos, source, cell, src, dst, sh, torch::kCPU, -1.0, &me_u);
  cfg.damping = true; solver.set_config(cfg);
  double Ed = solver.mbd_energy_dense(pos, source, cell, src, dst, sh, torch::kCPU, -1.0, &me_d);
  std::printf("CPP_UNDAMPED min_eig=%.6f E=%.6f  (%s)\n", me_u, Eu, me_u < 0 ? "INDEFINITE-UNPHYSICAL" : "PD");
  std::printf("CPP_DAMPED   min_eig=%.6f E=%.6f  (%s)\n", me_d, Ed, me_d > 0 ? "POSITIVE-DEFINITE-OK" : "INDEFINITE");
  return 0;
}
#endif
