#include "pair_mff_torch_kokkos.h"

#ifdef LMP_KOKKOS

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "neigh_list.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "neighbor_kokkos.h"

#include "mff_torch_engine.h"
#include "mff_device_runtime.h"
#include "mff_reciprocal_solver.h"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <c10/core/InferenceMode.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <unordered_map>
#include <vector>

using namespace LAMMPS_NS;

namespace {

int kokkos_gpu_device() {
#ifdef KOKKOS_ENABLE_HIP
  return Kokkos::HIP().hip_device();
#else
  return Kokkos::Cuda().cuda_device();
#endif
}

c10::Stream kokkos_torch_stream(int device) {
#ifdef KOKKOS_ENABLE_HIP
  return c10::hip::getStreamFromExternalMasqueradingAsCUDA(Kokkos::HIP().hip_stream(), device);
#else
  return c10::cuda::getStreamFromExternal(Kokkos::Cuda().cuda_stream(), device);
#endif
}

struct CellGeom {
  float cell[3][3];
  float inv[3][3];
  int pbc[3];
};

CellGeom build_cell_geom(const LAMMPS_NS::Domain* domain) {
  CellGeom g{};
  g.cell[0][0] = static_cast<float>(domain->xprd);
  g.cell[0][1] = 0.0f;
  g.cell[0][2] = 0.0f;
  g.cell[1][0] = static_cast<float>(domain->xy);
  g.cell[1][1] = static_cast<float>(domain->yprd);
  g.cell[1][2] = 0.0f;
  g.cell[2][0] = static_cast<float>(domain->xz);
  g.cell[2][1] = static_cast<float>(domain->yz);
  g.cell[2][2] = static_cast<float>(domain->zprd);
  g.pbc[0] = domain->xperiodic;
  g.pbc[1] = domain->yperiodic;
  g.pbc[2] = domain->zperiodic;

  const double a = g.cell[0][0], b = g.cell[0][1], c = g.cell[0][2];
  const double d = g.cell[1][0], e = g.cell[1][1], f = g.cell[1][2];
  const double h = g.cell[2][0], i = g.cell[2][1], j = g.cell[2][2];
  const double det = a * (e * j - f * i) - b * (d * j - f * h) + c * (d * i - e * h);
  if (std::abs(det) < 1e-12) {
    throw std::runtime_error("mff/torch/kk encountered a singular cell matrix");
  }
  const double inv_det = 1.0 / det;
  g.inv[0][0] = static_cast<float>((e * j - f * i) * inv_det);
  g.inv[0][1] = static_cast<float>((c * i - b * j) * inv_det);
  g.inv[0][2] = static_cast<float>((b * f - c * e) * inv_det);
  g.inv[1][0] = static_cast<float>((f * h - d * j) * inv_det);
  g.inv[1][1] = static_cast<float>((a * j - c * h) * inv_det);
  g.inv[1][2] = static_cast<float>((c * d - a * f) * inv_det);
  g.inv[2][0] = static_cast<float>((d * i - e * h) * inv_det);
  g.inv[2][1] = static_cast<float>((b * h - a * i) * inv_det);
  g.inv[2][2] = static_cast<float>((a * e - b * d) * inv_det);
  return g;
}

KOKKOS_INLINE_FUNCTION
int nearest_int_device(const float x) {
  return (x >= 0.0f) ? static_cast<int>(x + 0.5f) : static_cast<int>(x - 0.5f);
}

KOKKOS_INLINE_FUNCTION
bool lexicographic_positive_shift(const int sx, const int sy, const int sz) {
  return sx > 0 || (sx == 0 && sy > 0) || (sx == 0 && sy == 0 && sz > 0);
}

KOKKOS_INLINE_FUNCTION
bool keep_canonical_mbd_edge(const int src, const int dst, const int sx, const int sy, const int sz) {
  return src < dst || (src == dst && lexicographic_positive_shift(sx, sy, sz));
}

double norm3_host(double x, double y, double z) {
  return std::sqrt(x * x + y * y + z * z);
}

double periodic_face_height(const CellGeom& geom, int axis) {
  const double ax = geom.cell[0][0], ay = geom.cell[0][1], az = geom.cell[0][2];
  const double bx = geom.cell[1][0], by = geom.cell[1][1], bz = geom.cell[1][2];
  const double cx = geom.cell[2][0], cy = geom.cell[2][1], cz = geom.cell[2][2];
  const double det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
  const double volume = std::abs(det);
  if (axis == 0) return volume / std::max(norm3_host(by * cz - bz * cy, bz * cx - bx * cz, bx * cy - by * cx), 1.0e-12);
  if (axis == 1) return volume / std::max(norm3_host(cy * az - cz * ay, cz * ax - cx * az, cx * ay - cy * ax), 1.0e-12);
  return volume / std::max(norm3_host(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx), 1.0e-12);
}

void validate_mbd_dispersion_single_image_cutoff(
    Error *error, const CellGeom& geom, double dispersion_cutoff, const char *style_name) {
  const double tol = 1.0e-9 * std::max(1.0, dispersion_cutoff);
  for (int axis = 0; axis < 3; ++axis) {
    if (!geom.pbc[axis]) continue;
    const double height = periodic_face_height(geom, axis);
    if (2.0 * dispersion_cutoff > height + tol) {
      error->all(
          FLERR,
          (std::string(style_name) +
           " MBD dispersion cutoff is too large for the runtime nearest-image dispersion graph: "
           "2*dispersion_cutoff=" + std::to_string(2.0 * dispersion_cutoff) +
           " exceeds periodic face height " + std::to_string(height) +
           ". LAMMPS mff/torch deployment cannot represent the exact multi-image/self-image "
           "MBD graph used by the Python brute-force small-cell path; use a larger cell, a smaller "
           "dispersion cutoff, or train/deploy with mbd_operator_backend=pme_fft.")
              .c_str());
    }
  }
}

void sort_edge_vectors(std::vector<int64_t>& src, std::vector<int64_t>& dst, std::vector<float>& shifts) {
  const size_t n = src.size();
  if (n <= 1) return;
  std::vector<size_t> order(n);
  for (size_t i = 0; i < n; ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    if (dst[a] != dst[b]) return dst[a] < dst[b];
    if (src[a] != src[b]) return src[a] < src[b];
    for (int k = 0; k < 3; ++k) {
      const float sa = shifts[3 * a + static_cast<size_t>(k)];
      const float sb = shifts[3 * b + static_cast<size_t>(k)];
      if (sa != sb) return sa < sb;
    }
    return a < b;
  });
  std::vector<int64_t> src_sorted;
  std::vector<int64_t> dst_sorted;
  std::vector<float> shifts_sorted;
  src_sorted.reserve(n);
  dst_sorted.reserve(n);
  shifts_sorted.reserve(3 * n);
  for (size_t idx : order) {
    src_sorted.push_back(src[idx]);
    dst_sorted.push_back(dst[idx]);
    shifts_sorted.push_back(shifts[3 * idx + 0]);
    shifts_sorted.push_back(shifts[3 * idx + 1]);
    shifts_sorted.push_back(shifts[3 * idx + 2]);
  }
  src.swap(src_sorted);
  dst.swap(dst_sorted);
  shifts.swap(shifts_sorted);
}

mfftorch::ReciprocalInputs make_reciprocal_inputs(
    MPI_Comm world,
    const torch::Tensor& local_pos,
    const torch::Tensor& local_source,
    const torch::Tensor& cell,
    const CellGeom& geom,
    bool need_energy,
    const torch::Device& preferred_device) {
  mfftorch::ReciprocalInputs inputs;
  int world_rank = 0;
  int world_size = 1;
  MPI_Comm_rank(world, &world_rank);
  MPI_Comm_size(world, &world_size);
  int64_t local_n = local_pos.defined() ? local_pos.size(0) : 0;
  int64_t global_offset = 0;
  MPI_Exscan(&local_n, &global_offset, 1, MPI_LONG_LONG, MPI_SUM, world);
  if (world_rank == 0) global_offset = 0;
  inputs.world = world;
  inputs.local_pos = local_pos;
  inputs.local_source = local_source;
  inputs.local_global_ids =
      torch::arange(global_offset, global_offset + local_n, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  inputs.cell = cell;
  inputs.pbc = {geom.pbc[0], geom.pbc[1], geom.pbc[2]};
  inputs.need_energy = need_energy;
  inputs.preferred_device = preferred_device;
  inputs.world_rank = world_rank;
  inputs.world_size = world_size;
  return inputs;
}

}  // namespace

template <class DeviceType>
PairMFFTorchKokkos<DeviceType>::PairMFFTorchKokkos(LAMMPS *lmp) : PairMFFTorch(lmp) {
  kokkosable = 1;
  atomKK = (AtomKokkos *)atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  // VerletKokkos reads these through Pair*. Do not shadow execution_space:
  // marking stale host forces modified after compute discards device forces.
  datamask_read = X_MASK | F_MASK | TYPE_MASK | TAG_MASK;
  datamask_modify = F_MASK;
}

template <class DeviceType>
void PairMFFTorchKokkos<DeviceType>::init_style() {
  const bool debug_bundle = std::getenv("MFF_DEBUG_BUNDLE") != nullptr;
  if (core_pt_path_.empty()) error->all(FLERR, "pair_coeff for mff/torch must specify core.pt path");
  configure_message_depth();

  fold_mode_ = (comm->nprocs == 1);
  if (fold_mode_) {
    if (atom->map_style == Atom::MAP_NONE)
      error->all(FLERR, "pair_style mff/torch/kk (single-rank) needs an atom map to fold periodic ghosts "
                        "to local atoms; add 'atom_modify map yes' (or array/hash) to your input.");
    neighbor->add_request(this, NeighConst::REQ_FULL);
  } else {
    neighbor->add_request(this, layerwise_exchange_ ? NeighConst::REQ_FULL :
                          NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
    const double halo = std::max(static_cast<double>(layerwise_exchange_ ? 1 : mp_depth_) * cut_global_, request_cut_global_);
    if (comm->cutghostuser < halo) comm->cutghostuser = halo;
  }

  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType, LMPHostType> && !std::is_same_v<DeviceType, LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType, LMPDeviceType>);
  if (neighflag == FULL) request->enable_full();

  if (!engine_) engine_ = std::make_unique<mfftorch::MFFTorchEngine>();
  if (!engine_loaded_) {
    // Kokkos resolves MPI/launcher affinity. Use its actual device, since rank
    // modulo ngpus can disagree with SLURM_LOCALID or an explicit -k device.
    std::string dev = device_str_;
    if (dev == "cuda") {
      const int gpu_id=kokkos_gpu_device();
      dev = "cuda:" + std::to_string(gpu_id);
    }
    if(debug_bundle)std::fprintf(stderr,"[USER-MFFTORCH] rank=%d Kokkos device=%d Torch device=%s\n",
        comm->me,kokkos_gpu_device(),dev.c_str());
    try {
      if (!reciprocal_solver_) reciprocal_solver_ = std::make_unique<mfftorch::MFFReciprocalSolver>();
      if (debug_bundle) std::fprintf(stderr, "[USER-MFFTORCH] kk init_style before load_core\n");
      engine_->load_core(core_pt_path_, dev);
      if (debug_bundle) std::fprintf(stderr, "[USER-MFFTORCH] kk init_style after load_core\n");
      if (reciprocal_solver_) {
        auto cfg = reciprocal_solver_->config();
        // Mirror the non-Kokkos PairMFFTorch reciprocal config so the GPU path decodes the packed
        // latent-multipole source correctly (source_channels/max_multipole_l) at the trained mesh.
        cfg.mesh_size = static_cast<int>(engine_->long_range_mesh_size());
        cfg.max_multipole_l = static_cast<int>(engine_->long_range_max_multipole_l());
        cfg.source_channels = static_cast<int>(engine_->reciprocal_source_channels());
        cfg.slab_padding_factor = static_cast<int>(engine_->reciprocal_source_slab_padding_factor());
        cfg.green_mode = (engine_->long_range_green_mode() == "learned_poisson")
                             ? mfftorch::ReciprocalGreenMode::LearnedPoisson
                             : mfftorch::ReciprocalGreenMode::Poisson;
        // Latent-multipole alignment with the in-model multipole_energy (screening + energy scale).
        cfg.full_ewald = engine_->long_range_mesh_fft_full_ewald();
        cfg.ewald_alpha_prefactor = engine_->long_range_ewald_alpha_prefactor();
        cfg.energy_scale = engine_->long_range_energy_scale();
        cfg.assignment = (engine_->long_range_assignment() == "pcs") ? 1 : 0;  // match in-model PME
        reciprocal_solver_->set_config(cfg);
      }

      validate_external_field_configuration();
      if (debug_bundle) std::fprintf(stderr, "[USER-MFFTORCH] kk init_style after validate\n");
      engine_loaded_ = true;
    } catch (const std::exception &e) {
      error->all(FLERR, (std::string("Failed to load TorchScript core: ") + e.what()).c_str());
    }
  }

  if (engine_->is_cuda()) {
    if (debug_bundle) std::fprintf(stderr, "[USER-MFFTORCH] kk init_style before initial warmup\n");
    auto t = torch::from_blob(type2Z_.data(), {(int64_t)type2Z_.size()},
                              torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
                 .clone();
    type2Z_cuda_ = t.to(engine_->device());

    engine_->warmup(32, 256);
    if (debug_bundle) std::fprintf(stderr, "[USER-MFFTORCH] kk init_style after initial warmup\n");
  }
}

template <class DeviceType>
void PairMFFTorchKokkos<DeviceType>::compute(int eflag_in, int vflag_in) {
  const bool debug_kk_timings = []() {
    const char* env = std::getenv("MFF_DEBUG_KK_TIMINGS");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  const auto t_total_start = std::chrono::steady_clock::now();
  auto t_last = t_total_start;
  double edge_count_ms = 0.0;
  double offset_scan_ms = 0.0;
  double fill_ms = 0.0;
  double prepare_ms = 0.0;
  double compute_ms = 0.0;
  double reciprocal_ms = 0.0;
  double force_ms = 0.0;
  double atom_output_ms = 0.0;
  double virial_ms = 0.0;
  auto finish_segment = [&](double& bucket) {
    if (!debug_kk_timings) return;
    Kokkos::fence();
    const auto now = std::chrono::steady_clock::now();
    bucket += std::chrono::duration<double, std::milli>(now - t_last).count();
    t_last = now;
  };

	  if (!engine_loaded_) init_style();
	  if (!engine_ || !engine_->is_cuda()) {
	    atomKK->sync(Host, ALL_MASK);
	    PairMFFTorch::compute(eflag_in, vflag_in);
	    atomKK->modified(Host, F_MASK);
	    atomKK->sync(execution_space, F_MASK);
	    return;
	  }

	  int eflag = eflag_in;
  int vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;
  // Use default allocation behavior so eatom/vatom are available when
  // computes like pe/atom or stress/atom request per-atom quantities.
  ev_init(eflag, vflag);
  reset_physical_outputs();

  const bool fold = fold_mode_;
  atomKK->sync(execution_space, X_MASK | F_MASK | TYPE_MASK | (fold ? TAG_MASK : 0));
  const int map_style = atom->map_style;
  auto k_map_array = atomKK->k_map_array;
  auto k_map_hash = atomKK->k_map_hash;
  if (fold) {
    if (map_style == Atom::MAP_ARRAY) {
      k_map_array.template sync<DeviceType>();
    } else if (map_style == Atom::MAP_HASH) {
      k_map_hash.template sync<DeviceType>();
    }
  }

  auto x = atomKK->k_x.template view<DeviceType>();
  auto f = atomKK->k_f.template view<DeviceType>();
  auto type = atomKK->k_type.template view<DeviceType>();
  auto tag = atomKK->k_tag.template view<DeviceType>();

  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  const int nlocal_owned = nlocal;
  const int ntotal = nall;
  refresh_domain_owners();
  const int nmodel = fold ? nlocal_owned : ntotal;

  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;
	  const int inum = list->inum + (fold_mode_ ? 0 : list->gnum);
	  auto dev = engine_->device();
  // All Torch packing, sorting, model work, and Kokkos force writes share one
  // device stream. A guard only around compute leaves packing/force races.
  c10::DeviceGuard device_guard(dev);
  c10::StreamGuard stream_guard(kokkos_torch_stream(dev.index()));
		  const CellGeom geom = build_cell_geom(domain);
		  if (engine_->requires_mbd_dispersion_edges()) {
		    validate_mbd_dispersion_single_image_cutoff(
		        error, geom, dispersion_cut_global_, "pair_style mff/torch/kk");
		  }
		  const bool host_stage_aoti_dispersion =
	      fold && engine_->is_aoti_mode() && engine_->aoti_takes_dispersion_edges() && dispersion_cut_global_ > 0.0;
	  if (host_stage_aoti_dispersion) {
	    Kokkos::fence();
	    atomKK->sync(Host, X_MASK | F_MASK | TYPE_MASK | TAG_MASK);
	    auto h_ilist = Kokkos::create_mirror_view_and_copy(LMPHostType(), k_list->d_ilist);
	    auto h_numneigh = Kokkos::create_mirror_view_and_copy(LMPHostType(), k_list->d_numneigh);
	    auto h_neighbors = Kokkos::create_mirror_view_and_copy(LMPHostType(), k_list->d_neighbors);

	    double **x_host = atom->x;
	    double **f_host = atom->f;
	    int *type_host = atom->type;
	    tagint *tag_host = atom->tag;
	    const int nmodel = nlocal_owned;

	    std::unordered_map<tagint, int> local_owner_by_tag;
	    local_owner_by_tag.reserve(static_cast<size_t>(std::max(nlocal_owned, 0)));
	    for (int i = 0; i < nlocal_owned; ++i) local_owner_by_tag[tag_host[i]] = i;

	    buf_A_cpu_.resize(static_cast<size_t>(nmodel));
	    buf_pos_cpu_.resize(static_cast<size_t>(nmodel) * 3);
	    for (int i = 0; i < nmodel; ++i) {
	      const int itype = type_host[i];
	      buf_A_cpu_[i] = (itype >= 0 && itype < static_cast<int>(type2Z_.size())) ? type2Z_[itype] : 0;
	      buf_pos_cpu_[static_cast<size_t>(i) * 3 + 0] = static_cast<float>(x_host[i][0]);
	      buf_pos_cpu_[static_cast<size_t>(i) * 3 + 1] = static_cast<float>(x_host[i][1]);
	      buf_pos_cpu_[static_cast<size_t>(i) * 3 + 2] = static_cast<float>(x_host[i][2]);
	    }

	    int64_t Emax = 0;
	    for (int ii = 0; ii < inum; ++ii) Emax += h_numneigh(h_ilist(ii));
	    buf_edge_src_cpu_.clear();
	    buf_edge_dst_cpu_.clear();
	    buf_edge_shifts_cpu_.clear();
	    buf_disp_edge_src_cpu_.clear();
	    buf_disp_edge_dst_cpu_.clear();
	    buf_disp_edge_shifts_cpu_.clear();
	    buf_edge_src_cpu_.reserve(static_cast<size_t>(Emax));
	    buf_edge_dst_cpu_.reserve(static_cast<size_t>(Emax));
	    buf_edge_shifts_cpu_.reserve(static_cast<size_t>(Emax) * 3);
	    buf_disp_edge_src_cpu_.reserve(static_cast<size_t>(Emax));
	    buf_disp_edge_dst_cpu_.reserve(static_cast<size_t>(Emax));
	    buf_disp_edge_shifts_cpu_.reserve(static_cast<size_t>(Emax) * 3);

	    auto nearest_int_host = [](double x) {
	      return (x >= 0.0) ? static_cast<int>(x + 0.5) : static_cast<int>(x - 0.5);
	    };
	    for (int ii = 0; ii < inum; ++ii) {
	      const int i = h_ilist(ii);
	      const int jnum = h_numneigh(i);
	      for (int jj = 0; jj < jnum; ++jj) {
	        const int j = h_neighbors(i, jj) & NEIGHMASK;
	        const double rawx = x_host[j][0] - x_host[i][0];
	        const double rawy = x_host[j][1] - x_host[i][1];
	        const double rawz = x_host[j][2] - x_host[i][2];
	        const int sx = 0, sy = 0, sz = 0;
	        const double delx = rawx + sx * geom.cell[0][0] + sy * geom.cell[1][0] + sz * geom.cell[2][0];
	        const double dely = rawy + sx * geom.cell[0][1] + sy * geom.cell[1][1] + sz * geom.cell[2][1];
	        const double delz = rawz + sx * geom.cell[0][2] + sy * geom.cell[1][2] + sz * geom.cell[2][2];
	        const double rsq = delx * delx + dely * dely + delz * delz;
	        if (rsq > cutsq_global_ && rsq > dispersion_cutsq_global_) continue;
	        const auto owner_it = local_owner_by_tag.find(tag_host[j]);
	        if (owner_it == local_owner_by_tag.end()) continue;
	        const int jl = owner_it->second;
	        const double dxl = x_host[j][0] - x_host[jl][0];
	        const double dyl = x_host[j][1] - x_host[jl][1];
	        const double dzl = x_host[j][2] - x_host[jl][2];
	        const int gx = nearest_int_host(dxl * geom.inv[0][0] + dyl * geom.inv[1][0] + dzl * geom.inv[2][0]);
	        const int gy = nearest_int_host(dxl * geom.inv[0][1] + dyl * geom.inv[1][1] + dzl * geom.inv[2][1]);
	        const int gz = nearest_int_host(dxl * geom.inv[0][2] + dyl * geom.inv[1][2] + dzl * geom.inv[2][2]);
	        const int out_sx = -(gx + sx);
	        const int out_sy = -(gy + sy);
	        const int out_sz = -(gz + sz);
	        if (rsq <= cutsq_global_) {
	          buf_edge_src_cpu_.push_back(static_cast<int64_t>(jl));
	          buf_edge_dst_cpu_.push_back(static_cast<int64_t>(i));
	          buf_edge_shifts_cpu_.push_back(static_cast<float>(out_sx));
	          buf_edge_shifts_cpu_.push_back(static_cast<float>(out_sy));
	          buf_edge_shifts_cpu_.push_back(static_cast<float>(out_sz));
	        }
	        if (rsq <= dispersion_cutsq_global_ &&
	            keep_canonical_mbd_edge(jl, i, out_sx, out_sy, out_sz)) {
	          buf_disp_edge_src_cpu_.push_back(static_cast<int64_t>(jl));
	          buf_disp_edge_dst_cpu_.push_back(static_cast<int64_t>(i));
	          buf_disp_edge_shifts_cpu_.push_back(static_cast<float>(out_sx));
	          buf_disp_edge_shifts_cpu_.push_back(static_cast<float>(out_sy));
	          buf_disp_edge_shifts_cpu_.push_back(static_cast<float>(out_sz));
	        }
	      }
	    }

	    const int64_t E = static_cast<int64_t>(buf_edge_src_cpu_.size());
	    const int64_t Edisp = static_cast<int64_t>(buf_disp_edge_src_cpu_.size());
	    if (E == 0 && Edisp == 0) return;
	    sort_edge_vectors(buf_edge_src_cpu_, buf_edge_dst_cpu_, buf_edge_shifts_cpu_);
	    sort_edge_vectors(buf_disp_edge_src_cpu_, buf_disp_edge_dst_cpu_, buf_disp_edge_shifts_cpu_);
	    auto make_i64 = [](const std::vector<int64_t> &v) {
	      auto t = torch::empty({static_cast<int64_t>(v.size())}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
	      if (!v.empty()) std::memcpy(t.data_ptr<int64_t>(), v.data(), v.size() * sizeof(int64_t));
	      return t;
	    };
	    auto make_f32_2d = [](const std::vector<float> &v) {
	      auto t = torch::empty({static_cast<int64_t>(v.size() / 3), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
	      if (!v.empty()) std::memcpy(t.data_ptr<float>(), v.data(), v.size() * sizeof(float));
	      return t;
	    };
	    auto pos_t = make_f32_2d(buf_pos_cpu_);
	    auto A_t = make_i64(buf_A_cpu_);
	    auto edge_src_t = make_i64(buf_edge_src_cpu_);
	    auto edge_dst_t = make_i64(buf_edge_dst_cpu_);
	    auto edge_shifts_t = make_f32_2d(buf_edge_shifts_cpu_);
	    auto disp_edge_src_t = make_i64(buf_disp_edge_src_cpu_);
	    auto disp_edge_dst_t = make_i64(buf_disp_edge_dst_cpu_);
	    auto disp_edge_shifts_t = make_f32_2d(buf_disp_edge_shifts_cpu_);
	    const float cell_cpu[9] = {
	        geom.cell[0][0], geom.cell[0][1], geom.cell[0][2],
	        geom.cell[1][0], geom.cell[1][1], geom.cell[1][2],
	        geom.cell[2][0], geom.cell[2][1], geom.cell[2][2],
	    };
	    auto cell_t = torch::from_blob(
	                      const_cast<float *>(cell_cpu),
	                      {1, 3, 3},
	                      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
	                      .clone();
	    mfftorch::MFFOutputs out;
	    try {
	      out = engine_->compute(nlocal_owned, nmodel, pos_t, A_t, edge_src_t, edge_dst_t, edge_shifts_t, cell_t,
	                             disp_edge_src_t, disp_edge_dst_t, disp_edge_shifts_t,
	                             current_external_tensor(torch::kCPU), current_fidelity_tensor(torch::kCPU),
	                             static_cast<bool>(eflag_global || eflag_atom), static_cast<bool>(vflag_atom));
	    } catch (const std::exception &e) {
	      if(layerwise_exchange_ && comm->nprocs>1)
      error->one(FLERR, (std::string("mff/torch/kk host-staged engine compute failed: ") + e.what()).c_str());
    else error->all(FLERR, (std::string("mff/torch/kk host-staged engine compute failed: ") + e.what()).c_str());
	    }
	    mfftorch::ReciprocalOutputs reciprocal_out;
	    const bool exports_runtime_source =
	        engine_->exports_reciprocal_source() && engine_->reciprocal_source_channels() > 0;

	    const bool use_reciprocal =
	        reciprocal_solver_ && exports_runtime_source;
	    const bool use_runtime_long_range = use_reciprocal;
	    if (use_reciprocal) {
	      try {
	        auto local_source = out.reciprocal_source.defined()
	                                ? out.reciprocal_source.narrow(0, 0, nlocal_owned).to(dev, torch::kFloat32).contiguous()
	                                : torch::zeros(
	                                      {nlocal_owned, engine_->reciprocal_source_channels()},
	                                      torch::TensorOptions().dtype(torch::kFloat32).device(dev));

	        auto reciprocal_inputs = make_reciprocal_inputs(
	            world,
	            pos_t.to(dev, torch::kFloat32).contiguous(),
	            local_source,
	            cell_t.to(dev, torch::kFloat32).contiguous(),
	            geom,
	            static_cast<bool>(eflag_global || eflag_atom),
	            dev);
	        if (std::getenv("MFF_RECIPROCAL_MULTIPOLE_AUTOGRAD")) {
	          reciprocal_out = reciprocal_solver_->compute(reciprocal_inputs);
	        } else {
	          c10::InferenceMode reciprocal_inference_guard(true);
	          reciprocal_out = reciprocal_solver_->compute(reciprocal_inputs);
	        }
	      } catch (const std::exception &e) {
	        error->all(FLERR, (std::string("mff/torch/kk host-staged runtime long-range solver failed: ") + e.what()).c_str());
	      }
	    }
	    cache_physical_outputs(out, nlocal_owned);
	    if (eflag_global) eng_vdwl += out.energy;
	    if (eflag_global && use_runtime_long_range) eng_vdwl += reciprocal_out.energy;
	    auto forces_cpu = out.forces.to(torch::kCPU, torch::kFloat64).contiguous();
	    const double *force_ptr = forces_cpu.data_ptr<double>();
	    for (int i = 0; i < nlocal_owned; ++i) {
	      const double fx = force_ptr[static_cast<size_t>(i) * 3 + 0];
	      const double fy = force_ptr[static_cast<size_t>(i) * 3 + 1];
	      const double fz = force_ptr[static_cast<size_t>(i) * 3 + 2];
	      f_host[i][0] += fx;
	      f_host[i][1] += fy;
	      f_host[i][2] += fz;
#ifdef MFF_ENABLE_VIRIAL
	      if (vflag_global) {
	        virial[0] += fx * x_host[i][0];
	        virial[1] += fy * x_host[i][1];
	        virial[2] += fz * x_host[i][2];
	        virial[3] += fy * x_host[i][0];
	        virial[4] += fz * x_host[i][0];
	        virial[5] += fz * x_host[i][1];
	      }
#endif
	    }
	    if (use_runtime_long_range && reciprocal_out.forces_local.defined()) {
	      auto reciprocal_forces_cpu = reciprocal_out.forces_local.to(torch::kCPU, torch::kFloat64).contiguous();
	      const double *rfp = reciprocal_forces_cpu.data_ptr<double>();
	      for (int i = 0; i < nlocal_owned; ++i) {
	        const double fx = rfp[static_cast<size_t>(i) * 3 + 0];
	        const double fy = rfp[static_cast<size_t>(i) * 3 + 1];
	        const double fz = rfp[static_cast<size_t>(i) * 3 + 2];
	        f_host[i][0] += fx;
	        f_host[i][1] += fy;
	        f_host[i][2] += fz;
#ifdef MFF_ENABLE_VIRIAL
	        if (vflag_global) {
	          virial[0] += fx * x_host[i][0];
	          virial[1] += fy * x_host[i][1];
	          virial[2] += fz * x_host[i][2];
	          virial[3] += fy * x_host[i][0];
	          virial[4] += fz * x_host[i][0];
	          virial[5] += fz * x_host[i][1];
	        }
#endif
	      }
	    }
	    if (eflag_atom && out.atom_energy.defined()) {
	      auto ae_cpu = out.atom_energy.to(torch::kCPU, torch::kFloat64).contiguous().view({nmodel});
	      const double *ae = ae_cpu.data_ptr<double>();
	      for (int i = 0; i < nlocal_owned; ++i) eatom[i] += ae[i];
	    }
	    if (eflag_atom && use_runtime_long_range && reciprocal_out.atom_energy_local.defined()) {
	      auto ae_recip = reciprocal_out.atom_energy_local.to(torch::kCPU, torch::kFloat64).contiguous();
	      const double *ep = ae_recip.data_ptr<double>();
	      for (int i = 0; i < nlocal_owned; ++i) eatom[i] += ep[i];
	    }
	    atomKK->modified(Host, F_MASK);
	    if (vflag_fdotr) virial_fdotr_compute();
	    return;
	  }
	  if (nlocal == 0) {
    const std::array<float, 9> cell_values = {
        geom.cell[0][0], geom.cell[0][1], geom.cell[0][2],
        geom.cell[1][0], geom.cell[1][1], geom.cell[1][2],
        geom.cell[2][0], geom.cell[2][1], geom.cell[2][2],
    };
    if (!cached_cell_valid_ || cached_cell_values_ != cell_values || !cached_cell_t_.defined() ||
        cached_cell_t_.device() != dev) {
      cached_cell_values_ = cell_values;
      cached_cell_t_ = torch::from_blob(
                           cached_cell_values_.data(),
                           {1, 3, 3},
                           torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                           .clone()
                           .to(dev);
      cached_cell_valid_ = true;
    }
    auto cell_t = cached_cell_t_;

    const bool use_reciprocal =
        reciprocal_solver_ && engine_->exports_reciprocal_source() && engine_->reciprocal_source_channels() > 0;
    if (use_reciprocal) {
      auto empty_pos = torch::zeros({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
      auto empty_source = torch::zeros(
          {0, engine_->reciprocal_source_channels()},
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
      try {
        auto reciprocal_inputs = make_reciprocal_inputs(
            world,
            empty_pos,
            empty_source,
            cell_t.to(torch::kCPU, torch::kFloat32).contiguous(),
            geom,
            static_cast<bool>(eflag_global || eflag_atom),
            dev);

          (void)reciprocal_solver_->compute(reciprocal_inputs);

      } catch (const std::exception &e) {
        error->all(FLERR, (std::string("mff/torch/kk runtime long-range solver failed on empty rank: ") + e.what()).c_str());
      }
    }
    if(comm->nprocs>1 && !force->newton_pair) {
      reverse_ghost_contributions(torch::Tensor(),ntotal,false);
    }
    return;
  }
  if (inum == 0) return;

  // A K-layer local-energy graph needs incoming edges only for centers
  // reachable in fewer than K hops. Outer K-hop nodes are source-only.
  const int depth=layerwise_exchange_ ? 1 : mp_depth_;
  Kokkos::View<int *,DeviceType> center_level("mfftorch::center_level",ntotal);
  Kokkos::parallel_for("mfftorch::init_center_level",ntotal,KOKKOS_LAMBDA(const int i) {
    center_level(i)=i<nlocal_owned ? 0 : depth;
  });
  if(!fold) {
    const double cutoff2=cutsq_global_;
    for(int hop=0;hop<depth-1;++hop) {
      Kokkos::parallel_for("mfftorch::expand_centers",inum,KOKKOS_LAMBDA(const int ii) {
        const int i=d_ilist[ii];
        if(center_level(i)!=hop)return;
        for(int jj=0;jj<d_numneigh[i];++jj) {
          const int j=d_neighbors(i,jj)&NEIGHMASK;
          const double dx=x(j,0)-x(i,0),dy=x(j,1)-x(i,1),dz=x(j,2)-x(i,2);
          if(dx*dx+dy*dy+dz*dz<=cutoff2)Kokkos::atomic_min(&center_level(j),hop+1);
        }
      });
      Kokkos::fence();
    }
  }
  const bool include_dispersion=dispersion_cut_global_>0.0;
  // Count only required centers, rather than every halo ghost.
  int64_t Etotal = 0;
  Kokkos::parallel_reduce(
      "mfftorch::count_edges", inum,
      KOKKOS_LAMBDA(const int ii, int64_t &acc) {
        const int i=d_ilist[ii];
        if(center_level(i)<depth||include_dispersion)acc += static_cast<int64_t>(d_numneigh[i]);
      },
      Etotal);
  finish_segment(edge_count_ms);
  if (Etotal == 0 && !layerwise_exchange_) return;

  // Exclusive-scan offsets on device; reuse cached view when inum unchanged.
  if (cached_inum_ != inum) {
    cached_d_offsets_ = Kokkos::View<int64_t *, DeviceType>("mfftorch::offsets", inum);
    cached_inum_ = inum;
  }
  auto d_offsets = cached_d_offsets_;
  Kokkos::parallel_scan(
      "mfftorch::scan_offsets", inum,
      KOKKOS_LAMBDA(const int ii, int64_t &update, const bool is_final) {
        if (is_final) d_offsets(ii) = update;
        const int i=d_ilist[ii];
        if(center_level(i)<depth||include_dispersion)update += static_cast<int64_t>(d_numneigh[i]);
      });
  finish_segment(offset_scan_ms);

  // Reuse CUDA tensor buffers when edge count / atom count unchanged.
  using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;

  if (cached_Etotal_ != Etotal) {
    buf_edge_src_ = torch::empty({Etotal}, torch::TensorOptions().dtype(torch::kInt64).device(dev));
    buf_edge_dst_ = torch::empty({Etotal}, torch::TensorOptions().dtype(torch::kInt64).device(dev));
    buf_edge_shifts_ = torch::empty({Etotal, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    buf_disp_edge_src_ = torch::empty({Etotal}, torch::TensorOptions().dtype(torch::kInt64).device(dev));
    buf_disp_edge_dst_ = torch::empty({Etotal}, torch::TensorOptions().dtype(torch::kInt64).device(dev));
    buf_disp_edge_shifts_ = torch::empty({Etotal, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    cached_Etotal_ = Etotal;
  }
  if (cached_ntotal_ != static_cast<int64_t>(nmodel)) {
    buf_type_idx_ = torch::empty({nmodel}, torch::TensorOptions().dtype(torch::kInt64).device(dev));
    buf_pos_ = torch::empty({nmodel, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    cached_ntotal_ = nmodel;
  }

  Kokkos::View<int64_t *, DeviceType, Unmanaged> edge_src_v(buf_edge_src_.data_ptr<int64_t>(), Etotal);
  Kokkos::View<int64_t *, DeviceType, Unmanaged> edge_dst_v(buf_edge_dst_.data_ptr<int64_t>(), Etotal);
  Kokkos::View<int64_t *, DeviceType, Unmanaged> disp_edge_src_v(buf_disp_edge_src_.data_ptr<int64_t>(), Etotal);
  Kokkos::View<int64_t *, DeviceType, Unmanaged> disp_edge_dst_v(buf_disp_edge_dst_.data_ptr<int64_t>(), Etotal);
  Kokkos::View<int64_t *, DeviceType, Unmanaged> type_idx_v(buf_type_idx_.data_ptr<int64_t>(), nmodel);
  Kokkos::View<float **, Kokkos::LayoutRight, DeviceType, Unmanaged> edge_shifts_v(buf_edge_shifts_.data_ptr<float>(), Etotal, 3);
  Kokkos::View<float **, Kokkos::LayoutRight, DeviceType, Unmanaged> disp_edge_shifts_v(buf_disp_edge_shifts_.data_ptr<float>(), Etotal, 3);
  Kokkos::View<float **, Kokkos::LayoutRight, DeviceType, Unmanaged> pos_v(buf_pos_.data_ptr<float>(), nmodel, 3);

  const float i00 = geom.inv[0][0], i01 = geom.inv[0][1], i02 = geom.inv[0][2];
  const float i10 = geom.inv[1][0], i11 = geom.inv[1][1], i12 = geom.inv[1][2];
  const float i20 = geom.inv[2][0], i21 = geom.inv[2][1], i22 = geom.inv[2][2];
  const int px = geom.pbc[0], py = geom.pbc[1], pz = geom.pbc[2];
  const float c00 = geom.cell[0][0], c01 = geom.cell[0][1], c02 = geom.cell[0][2];
  const float c10 = geom.cell[1][0], c11 = geom.cell[1][1], c12 = geom.cell[1][2];
  const float c20 = geom.cell[2][0], c21 = geom.cell[2][1], c22 = geom.cell[2][2];
  const double cutsq = cutsq_global_;
  const bool use_dispersion_edges = dispersion_cut_global_ > 0.0;
  const double disp_cutsq = dispersion_cutsq_global_;

  Kokkos::parallel_for(
      "mfftorch::fill_pos_and_type", nmodel,
      KOKKOS_LAMBDA(const int i) {
        pos_v(i, 0) = static_cast<float>(x(i, 0));
        pos_v(i, 1) = static_cast<float>(x(i, 1));
        pos_v(i, 2) = static_cast<float>(x(i, 2));
        type_idx_v(i) = static_cast<int64_t>(type(i));
      });

	  Kokkos::parallel_for(
	      "mfftorch::fill_edges", inum, KOKKOS_LAMBDA(const int ii) {
        const int i = d_ilist[ii];
        if(center_level(i)>=depth && !include_dispersion)return;
        const int jnum = d_numneigh[i];
        const int64_t base = d_offsets(ii);
        for (int jj = 0; jj < jnum; jj++) {
          int j = d_neighbors(i, jj) & NEIGHMASK;
          const int64_t idx = base + jj;
          const double rawx = x(j, 0) - x(i, 0);
          const double rawy = x(j, 1) - x(i, 1);
          const double rawz = x(j, 2) - x(i, 2);
          const float fracx = rawx * i00 + rawy * i10 + rawz * i20;
          const float fracy = rawx * i01 + rawy * i11 + rawz * i21;
          const float fracz = rawx * i02 + rawy * i12 + rawz * i22;
          // LAMMPS ghosts already identify the physical periodic image.
          // Applying minimum-image wrapping again duplicates short-box edges.
          const int sx = 0;
          const int sy = 0;
          const int sz = 0;
          const float shiftx = sx * c00 + sy * c10 + sz * c20;
          const float shifty = sx * c01 + sy * c11 + sz * c21;
          const float shiftz = sx * c02 + sy * c12 + sz * c22;
          const double delx = rawx + shiftx;
          const double dely = rawy + shifty;
          const double delz = rawz + shiftz;
          const double rsq = delx * delx + dely * dely + delz * delz;
          const bool keep_main = center_level(i)<depth && rsq <= cutsq;
          const bool keep_disp_raw = use_dispersion_edges && rsq <= disp_cutsq;
          if (!keep_main && !keep_disp_raw) {
            edge_src_v(idx) = static_cast<int64_t>(-1);
            edge_dst_v(idx) = static_cast<int64_t>(-1);
            edge_shifts_v(idx, 0) = 0.0f;
            edge_shifts_v(idx, 1) = 0.0f;
            edge_shifts_v(idx, 2) = 0.0f;
            disp_edge_src_v(idx) = static_cast<int64_t>(-1);
            disp_edge_dst_v(idx) = static_cast<int64_t>(-1);
            disp_edge_shifts_v(idx, 0) = 0.0f;
            disp_edge_shifts_v(idx, 1) = 0.0f;
            disp_edge_shifts_v(idx, 2) = 0.0f;
            continue;
          }
          int src = j;
          int out_sx = sx;
          int out_sy = sy;
          int out_sz = sz;
          if (fold) {
            const auto wanted_tag = tag(j);
            src = AtomKokkos::map_kokkos<DeviceType>(tag(j), map_style, k_map_array, k_map_hash);
            if (src < 0 || src >= nlocal_owned || tag(src) != wanted_tag) {
              int found = -1;
              for (int k = 0; k < nlocal_owned; ++k) {
                if (tag(k) == wanted_tag) {
                  found = k;
                  break;
                }
              }
              src = found;
            }
            if (src < 0 || src >= nlocal_owned) {
              edge_src_v(idx) = static_cast<int64_t>(-1);
              edge_dst_v(idx) = static_cast<int64_t>(-1);
              edge_shifts_v(idx, 0) = 0.0f;
              edge_shifts_v(idx, 1) = 0.0f;
              edge_shifts_v(idx, 2) = 0.0f;
              disp_edge_src_v(idx) = static_cast<int64_t>(-1);
              disp_edge_dst_v(idx) = static_cast<int64_t>(-1);
              disp_edge_shifts_v(idx, 0) = 0.0f;
              disp_edge_shifts_v(idx, 1) = 0.0f;
              disp_edge_shifts_v(idx, 2) = 0.0f;
              continue;
            }
            const float owner_dx = static_cast<float>(x(j, 0) - x(src, 0));
            const float owner_dy = static_cast<float>(x(j, 1) - x(src, 1));
            const float owner_dz = static_cast<float>(x(j, 2) - x(src, 2));
            out_sx = nearest_int_device(owner_dx * i00 + owner_dy * i10 + owner_dz * i20) + sx;
            out_sy = nearest_int_device(owner_dx * i01 + owner_dy * i11 + owner_dz * i21) + sy;
            out_sz = nearest_int_device(owner_dx * i02 + owner_dy * i12 + owner_dz * i22) + sz;
          }
          // Match the CPU pair builder and model convention: edge_src is the
          // neighbor/owner and edge_dst is the center that receives the message.
          if (keep_main) {
            edge_src_v(idx) = static_cast<int64_t>(src);
            edge_dst_v(idx) = static_cast<int64_t>(i);
            edge_shifts_v(idx, 0) = static_cast<float>(-out_sx);
            edge_shifts_v(idx, 1) = static_cast<float>(-out_sy);
            edge_shifts_v(idx, 2) = static_cast<float>(-out_sz);
          } else {
            edge_src_v(idx) = static_cast<int64_t>(-1);
            edge_dst_v(idx) = static_cast<int64_t>(-1);
            edge_shifts_v(idx, 0) = 0.0f;
            edge_shifts_v(idx, 1) = 0.0f;
            edge_shifts_v(idx, 2) = 0.0f;
          }
          // (src, dst, shift) ~ (dst, src, -shift).  Self-image couplings keep
          // the lexicographically positive shift half, matching the Python builder.
          const bool keep_disp = keep_disp_raw && keep_canonical_mbd_edge(src, i, -out_sx, -out_sy, -out_sz);
          if (keep_disp) {
            disp_edge_src_v(idx) = static_cast<int64_t>(src);
            disp_edge_dst_v(idx) = static_cast<int64_t>(i);
            disp_edge_shifts_v(idx, 0) = static_cast<float>(-out_sx);
            disp_edge_shifts_v(idx, 1) = static_cast<float>(-out_sy);
            disp_edge_shifts_v(idx, 2) = static_cast<float>(-out_sz);
          } else {
            disp_edge_src_v(idx) = static_cast<int64_t>(-1);
            disp_edge_dst_v(idx) = static_cast<int64_t>(-1);
            disp_edge_shifts_v(idx, 0) = 0.0f;
            disp_edge_shifts_v(idx, 1) = 0.0f;
            disp_edge_shifts_v(idx, 2) = 0.0f;
          }
	        }
	      });
	  Kokkos::fence();
	  finish_segment(fill_ms);

  auto valid_mask = buf_edge_src_.ge(0);
  const int64_t Efiltered = valid_mask.sum().item<int64_t>();
  int64_t Edispfiltered = 0;
  torch::Tensor disp_valid_mask;
  if (use_dispersion_edges) {
    disp_valid_mask = buf_disp_edge_src_.ge(0);
    Edispfiltered = disp_valid_mask.sum().item<int64_t>();
  }
  if (!layerwise_exchange_ && Efiltered == 0 && (!use_dispersion_edges || Edispfiltered == 0)) return;
  if (Efiltered != Etotal) {
    auto valid_idx = torch::nonzero(valid_mask).view({-1});
    buf_edge_src_ = buf_edge_src_.index_select(0, valid_idx);
    buf_edge_dst_ = buf_edge_dst_.index_select(0, valid_idx);
    buf_edge_shifts_ = buf_edge_shifts_.index_select(0, valid_idx);
    cached_Etotal_ = Efiltered;
    Etotal = Efiltered;
  }
	  if (use_dispersion_edges && Edispfiltered != static_cast<int64_t>(buf_disp_edge_src_.size(0))) {
	    auto disp_valid_idx = torch::nonzero(disp_valid_mask).view({-1});
	    buf_disp_edge_src_ = buf_disp_edge_src_.index_select(0, disp_valid_idx);
	    buf_disp_edge_dst_ = buf_disp_edge_dst_.index_select(0, disp_valid_idx);
	    buf_disp_edge_shifts_ = buf_disp_edge_shifts_.index_select(0, disp_valid_idx);
	    cached_Etotal_ = -1;
	  }
	  auto sort_edges_by_dst_src_shift = [nmodel](torch::Tensor &src, torch::Tensor &dst, torch::Tensor &shifts) {
	    if (!src.defined() || src.numel() <= 1) return;
	    auto shift_i64 = shifts.to(torch::kInt64);
	    const auto max_abs = shift_i64.abs().max().item<int64_t>();
	    const int64_t base = 2 * max_abs + 1;
	    auto key = dst * static_cast<int64_t>(nmodel) + src;
	    key = key * base + (shift_i64.select(1, 0) + max_abs);
	    key = key * base + (shift_i64.select(1, 1) + max_abs);
	    key = key * base + (shift_i64.select(1, 2) + max_abs);
	    auto order = torch::argsort(key);
	    src = src.index_select(0, order).contiguous();
	    dst = dst.index_select(0, order).contiguous();
	    shifts = shifts.index_select(0, order).contiguous();
	  };
	  sort_edges_by_dst_src_shift(buf_edge_src_, buf_edge_dst_, buf_edge_shifts_);
	  if (use_dispersion_edges) {
	    sort_edges_by_dst_src_shift(buf_disp_edge_src_, buf_disp_edge_dst_, buf_disp_edge_shifts_);
	  }
	  if (std::getenv("MFF_VALIDATE_GRAPH") || std::getenv("MFF_DUMP_GRAPH")) {
    auto edge_src_cpu = buf_edge_src_.to(torch::kCPU, torch::kInt64).contiguous();
    auto edge_dst_cpu = buf_edge_dst_.to(torch::kCPU, torch::kInt64).contiguous();
    auto disp_src_cpu =
        use_dispersion_edges ? buf_disp_edge_src_.to(torch::kCPU, torch::kInt64).contiguous() : torch::Tensor();
    auto disp_dst_cpu =
        use_dispersion_edges ? buf_disp_edge_dst_.to(torch::kCPU, torch::kInt64).contiguous() : torch::Tensor();
    auto check_bounds = [&](const torch::Tensor &src_t, const torch::Tensor &dst_t, const char *label) {
      const int64_t *src = src_t.data_ptr<int64_t>();
      const int64_t *dst = dst_t.data_ptr<int64_t>();
      const int64_t ne = src_t.size(0);
      for (int64_t k = 0; k < ne; ++k) {
        if (src[k] < 0 || src[k] >= nmodel || dst[k] < 0 || dst[k] >= nmodel) {
          error->all(FLERR,
                     (std::string("mff/torch/kk built out-of-range ") + label + " edge at " +
                      std::to_string(k) + ": src=" + std::to_string(src[k]) +
                      " dst=" + std::to_string(dst[k]) + " nmodel=" + std::to_string(nmodel) +
                      " nlocal=" + std::to_string(nlocal) + " ntotal=" + std::to_string(ntotal))
                         .c_str());
        }
      }
    };
    if (std::getenv("MFF_VALIDATE_GRAPH")) {
      check_bounds(edge_src_cpu, edge_dst_cpu, "main");
      if (use_dispersion_edges) check_bounds(disp_src_cpu, disp_dst_cpu, "dispersion");
    }
	    if (std::getenv("MFF_DUMP_GRAPH")) {
	      std::fprintf(stderr, "[MFF_DUMP_GRAPH_KK] nmodel=%d nlocal=%d ntotal=%d E=%lld Edisp=%lld\n",
	                   nmodel, nlocal, ntotal, (long long)Etotal, (long long)Edispfiltered);
	      auto pos_cpu = buf_pos_.to(torch::kCPU, torch::kFloat32).contiguous();
	      auto A_cpu = type2Z_cuda_.index_select(0, buf_type_idx_).to(torch::kCPU, torch::kInt64).contiguous();
	      auto edge_shifts_cpu = buf_edge_shifts_.to(torch::kCPU, torch::kFloat32).contiguous();
	      auto disp_shifts_cpu =
	          use_dispersion_edges ? buf_disp_edge_shifts_.to(torch::kCPU, torch::kFloat32).contiguous() : torch::Tensor();
	      std::ofstream dbg("/tmp/mff_pair_edges_kk.txt");
	      dbg << "n_model " << nmodel << " nlocal " << nlocal << " ntotal " << ntotal
	          << " E " << Etotal << " Edisp " << Edispfiltered << "\n";
	      dbg << "atoms\n";
	      const auto *A_ptr = A_cpu.data_ptr<int64_t>();
	      const auto *pos_ptr = pos_cpu.data_ptr<float>();
	      for (int i = 0; i < nmodel; ++i) {
	        dbg << i << " " << A_ptr[i] << " "
	            << pos_ptr[static_cast<size_t>(i) * 3 + 0] << " "
	            << pos_ptr[static_cast<size_t>(i) * 3 + 1] << " "
	            << pos_ptr[static_cast<size_t>(i) * 3 + 2] << "\n";
	      }
	      dbg << "main\n";
	      const auto *src_ptr = edge_src_cpu.data_ptr<int64_t>();
	      const auto *dst_ptr = edge_dst_cpu.data_ptr<int64_t>();
	      const auto *shift_ptr = edge_shifts_cpu.data_ptr<float>();
	      for (int64_t k = 0; k < std::min<int64_t>(Etotal, 64); ++k) {
	        dbg << k << " " << src_ptr[k] << " " << dst_ptr[k] << " "
	            << shift_ptr[3 * k + 0] << " " << shift_ptr[3 * k + 1] << " "
	            << shift_ptr[3 * k + 2] << "\n";
	      }
	      dbg << "dispersion\n";
	      if (use_dispersion_edges) {
	        const auto *disp_src_ptr = disp_src_cpu.data_ptr<int64_t>();
	        const auto *disp_dst_ptr = disp_dst_cpu.data_ptr<int64_t>();
	        const auto *disp_shift_ptr = disp_shifts_cpu.data_ptr<float>();
	        for (int64_t k = 0; k < std::min<int64_t>(Edispfiltered, 64); ++k) {
	          dbg << k << " " << disp_src_ptr[k] << " " << disp_dst_ptr[k] << " "
	              << disp_shift_ptr[3 * k + 0] << " " << disp_shift_ptr[3 * k + 1] << " "
	              << disp_shift_ptr[3 * k + 2] << "\n";
	        }
	      }
	      auto print_minmax = [&](const torch::Tensor &src_t, const torch::Tensor &dst_t, const char *label) {
	        if (src_t.numel() == 0) {
	          std::fprintf(stderr, "[MFF_DUMP_GRAPH_KK] %s empty\n", label);
          return;
        }
        const auto smin = src_t.min().item<int64_t>();
        const auto smax = src_t.max().item<int64_t>();
        const auto dmin = dst_t.min().item<int64_t>();
        const auto dmax = dst_t.max().item<int64_t>();
        std::fprintf(stderr, "[MFF_DUMP_GRAPH_KK] %s src=[%lld,%lld] dst=[%lld,%lld]\n",
                     label, (long long)smin, (long long)smax, (long long)dmin, (long long)dmax);
      };
      print_minmax(edge_src_cpu, edge_dst_cpu, "main");
      if (use_dispersion_edges) print_minmax(disp_src_cpu, disp_dst_cpu, "dispersion");
    }
  }

  const bool need_energy = static_cast<bool>(eflag_global || eflag_atom);
  const bool need_atom_virial = static_cast<bool>(vflag_atom);
  mfftorch::MFFOutputs out;

  const std::array<float, 9> cell_values = {
      geom.cell[0][0], geom.cell[0][1], geom.cell[0][2],
      geom.cell[1][0], geom.cell[1][1], geom.cell[1][2],
      geom.cell[2][0], geom.cell[2][1], geom.cell[2][2],
  };
  if (!cached_cell_valid_ || cached_cell_values_ != cell_values || !cached_cell_t_.defined() ||
      cached_cell_t_.device() != dev) {
    cached_cell_values_ = cell_values;
    cached_cell_t_ = torch::from_blob(
                         cached_cell_values_.data(),
                         {1, 3, 3},
                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                         .clone()
                         .to(dev);
    cached_cell_valid_ = true;
  }
  auto cell_t = cached_cell_t_;

  try {
    Kokkos::fence();
    if (engine_->is_bundle_manifest() &&
        (prepared_nlocal_ != nlocal || prepared_ntotal_ != nmodel || prepared_nedges_ != Etotal)) {
      engine_->prepare_for_shape(nlocal, nmodel, Etotal);
      prepared_nlocal_ = nlocal;
      prepared_ntotal_ = nmodel;
      prepared_nedges_ = Etotal;
    }
    finish_segment(prepare_ms);
    const bool stage_for_aoti_dispersion =
        engine_->is_aoti_mode() && engine_->aoti_takes_dispersion_edges() && use_dispersion_edges;
    if (engine_->prefers_kokkos_host_staging() || stage_for_aoti_dispersion) {
      c10::StreamGuard default_stream_guard(mfftorch::gpu::default_stream(dev.index()));
      // Native cue custom CUDA ops are unstable when fed Kokkos-managed device
      // tensors and an external CUDA stream. Stage inputs through CPU so the
      // engine follows the same stable transfer path as plain mff/torch.
      auto pos_cpu = buf_pos_.to(torch::kCPU, torch::kFloat32).contiguous();
      auto A_cpu = type2Z_cuda_.index_select(0, buf_type_idx_).to(torch::kCPU, torch::kInt64).contiguous();
      auto edge_src_cpu = buf_edge_src_.to(torch::kCPU, torch::kInt64).contiguous();
      auto edge_dst_cpu = buf_edge_dst_.to(torch::kCPU, torch::kInt64).contiguous();
      auto edge_shifts_cpu = buf_edge_shifts_.to(torch::kCPU, torch::kFloat32).contiguous();
      auto disp_edge_src_cpu =
          use_dispersion_edges ? buf_disp_edge_src_.to(torch::kCPU, torch::kInt64).contiguous() : torch::Tensor();
      auto disp_edge_dst_cpu =
          use_dispersion_edges ? buf_disp_edge_dst_.to(torch::kCPU, torch::kInt64).contiguous() : torch::Tensor();
      auto disp_edge_shifts_cpu =
          use_dispersion_edges ? buf_disp_edge_shifts_.to(torch::kCPU, torch::kFloat32).contiguous() : torch::Tensor();
      auto external_tensor = current_external_tensor(torch::kCPU);
      auto fidelity_ids = current_fidelity_tensor(torch::kCPU);
      out = engine_->compute(
          nlocal,
          nmodel,
          pos_cpu,
          A_cpu,
          edge_src_cpu,
          edge_dst_cpu,
          edge_shifts_cpu,
          cell_t.to(torch::kCPU, torch::kFloat32).contiguous(),
          disp_edge_src_cpu,
          disp_edge_dst_cpu,
          disp_edge_shifts_cpu,
          external_tensor,
          fidelity_ids,
          need_energy,
          need_atom_virial);
      if (engine_->is_cuda()) torch::cuda::synchronize();
    } else {
      auto A = type2Z_cuda_.index_select(0, buf_type_idx_);
      auto external_tensor = current_external_tensor(dev);
      auto fidelity_ids = current_fidelity_tensor(dev);
      out = engine_->compute(nlocal, nmodel, buf_pos_, A, buf_edge_src_, buf_edge_dst_, buf_edge_shifts_, cell_t,
                             use_dispersion_edges ? buf_disp_edge_src_ : torch::Tensor(),
                             use_dispersion_edges ? buf_disp_edge_dst_ : torch::Tensor(),
                             use_dispersion_edges ? buf_disp_edge_shifts_ : torch::Tensor(),
                             external_tensor,
                             fidelity_ids,
                             need_energy, need_atom_virial);
    }
    finish_segment(compute_ms);
  } catch (const std::exception &e) {
    if(layerwise_exchange_ && comm->nprocs>1)
      error->one(FLERR, (std::string("mff/torch/kk engine compute failed: ") + e.what()).c_str());
    else error->all(FLERR, (std::string("mff/torch/kk engine compute failed: ") + e.what()).c_str());
  }
  cache_physical_outputs(out, nlocal);

  mfftorch::ReciprocalOutputs reciprocal_out;
  torch::Tensor reciprocal_forces_dev;
  const bool exports_runtime_source = engine_->exports_reciprocal_source() && engine_->reciprocal_source_channels() > 0;

  const bool use_reciprocal =
      reciprocal_solver_ && exports_runtime_source;
  const bool use_runtime_long_range = use_reciprocal;
  if (use_reciprocal) {
    try {
      auto local_source = out.reciprocal_source.defined()
                              ? out.reciprocal_source.narrow(0, 0, nlocal).contiguous()
                              : torch::zeros(
                                    {nlocal, engine_->reciprocal_source_channels()},
                                    torch::TensorOptions().dtype(torch::kFloat32).device(dev));

      auto reciprocal_inputs = make_reciprocal_inputs(
          world,
          buf_pos_.narrow(0, 0, nlocal).contiguous(),
          local_source,
          cell_t.contiguous(),
          geom,
          need_energy,
          dev);
      if (std::getenv("MFF_RECIPROCAL_MULTIPOLE_AUTOGRAD")) {
        reciprocal_out = reciprocal_solver_->compute(reciprocal_inputs);
      } else {
        c10::InferenceMode reciprocal_inference_guard(true);
        reciprocal_out = reciprocal_solver_->compute(reciprocal_inputs);
      }
    } catch (const std::exception &e) {
      error->all(FLERR, (std::string("mff/torch/kk runtime long-range solver failed: ") + e.what()).c_str());
    }
  }
  finish_segment(reciprocal_ms);

  if (eflag_global) eng_vdwl += out.energy;
  if (eflag_global && use_runtime_long_range) eng_vdwl += reciprocal_out.energy;

  // Write forces on device (no host transfer).
  auto forces = out.forces.contiguous();
  Kokkos::View<float **, Kokkos::LayoutRight, DeviceType, Unmanaged> forces_v(forces.data_ptr<float>(), nmodel, 3);

  const int nwrite = fold ? nlocal : (force->newton_pair ? ntotal : nlocal);
  Kokkos::parallel_for(
      "mfftorch::add_forces", nwrite, KOKKOS_LAMBDA(const int i) {
        f(i, 0) += forces_v(i, 0);
        f(i, 1) += forces_v(i, 1);
        f(i, 2) += forces_v(i, 2);
      });
  if (use_runtime_long_range && reciprocal_out.forces_local.defined()) {
    reciprocal_forces_dev = reciprocal_out.forces_local.to(dev, torch::kFloat32).contiguous();
    Kokkos::View<float **, Kokkos::LayoutRight, DeviceType, Unmanaged> reciprocal_forces_v(
        reciprocal_forces_dev.data_ptr<float>(), nlocal, 3);
    Kokkos::parallel_for(
        "mfftorch::add_reciprocal_forces", nlocal, KOKKOS_LAMBDA(const int i) {
          f(i, 0) += reciprocal_forces_v(i, 0);
          f(i, 1) += reciprocal_forces_v(i, 1);
          f(i, 2) += reciprocal_forces_v(i, 2);
        });
  }
  Kokkos::fence();
  atomKK->modified(execution_space, F_MASK);
  if(comm->nprocs>1 && !force->newton_pair) {
    reverse_ghost_contributions(out.forces,ntotal,false);
    auto received=torch::from_blob(reverse_force_buffer_.data(),{nlocal,3},
        torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)).to(dev,torch::kFloat32).contiguous();
    Kokkos::View<float **,Kokkos::LayoutRight,DeviceType,Unmanaged> received_v(received.template data_ptr<float>(),nlocal,3);
    Kokkos::parallel_for("mfftorch::add_received_ghost_forces",nlocal,KOKKOS_LAMBDA(const int i) {
      f(i,0)+=received_v(i,0);f(i,1)+=received_v(i,1);f(i,2)+=received_v(i,2);
    });
    Kokkos::fence();
    atomKK->modified(execution_space,F_MASK);
  }
  finish_segment(force_ms);

  // Per-atom energy: copy NN atom_energy to LAMMPS eatom (local atoms only).
  if (eflag_atom && eatom && out.atom_energy.defined()) {
    auto ae = out.atom_energy.to(torch::kCPU, torch::kFloat64).contiguous().view({nmodel});
    const double *ep = ae.data_ptr<double>();
    for (int i = 0; i < nlocal; i++) eatom[i] += ep[i];
  }
  if (eflag_atom && eatom && use_runtime_long_range && reciprocal_out.atom_energy_local.defined()) {
    auto ae_recip = reciprocal_out.atom_energy_local.to(torch::kCPU, torch::kFloat64).contiguous();
    const double *ep = ae_recip.data_ptr<double>();
    for (int i = 0; i < nlocal; i++) eatom[i] += ep[i];
  }
  finish_segment(atom_output_ms);

  // Per-atom virial: engine computed atom_virial [ntotal, 6] on GPU via edge-force
  // outer products (rij ⊗ edge_forces), scatter-added 50/50 to src and dst.
  // With newton OFF + FULL list, only write LOCAL atoms (consistent with ev_tally
  // convention — each pair is visited twice, so each local atom gets its full share).
  if (vflag_atom && vatom && out.atom_virial.defined()) {
    auto avir = out.atom_virial.to(torch::kCPU, torch::kFloat64).contiguous();
    const double *vp = avir.data_ptr<double>();
    const int nvir = fold ? nlocal : (force->newton_pair ? ntotal : nlocal);
    for (int i = 0; i < nvir; i++) {
      vatom[i][0] += vp[i * 6 + 0];
      vatom[i][1] += vp[i * 6 + 1];
      vatom[i][2] += vp[i * 6 + 2];
      vatom[i][3] += vp[i * 6 + 3];
      vatom[i][4] += vp[i * 6 + 4];
      vatom[i][5] += vp[i * 6 + 5];
    }
  }
  finish_segment(virial_ms);

#ifdef MFF_ENABLE_VIRIAL
  if (vflag_global) {
    // f·r virial must sum over ALL atoms (local + ghost).
    // Ghost atoms carry the "other half" of cross-boundary PBC interactions.
    // Using forces_v (NN output) directly guarantees ghost forces are included.
    Kokkos::View<double[6], DeviceType> d_virial("mfftorch::d_virial");
    Kokkos::deep_copy(d_virial, 0.0);

    Kokkos::parallel_for(
        "mfftorch::virial_fdotr", ntotal, KOKKOS_LAMBDA(const int i) {
          const double xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);
          const double fx = static_cast<double>(forces_v(i, 0));
          const double fy = static_cast<double>(forces_v(i, 1));
          const double fz = static_cast<double>(forces_v(i, 2));
          Kokkos::atomic_add(&d_virial(0), fx * xi);  // xx
          Kokkos::atomic_add(&d_virial(1), fy * yi);  // yy
          Kokkos::atomic_add(&d_virial(2), fz * zi);  // zz
          Kokkos::atomic_add(&d_virial(3), fy * xi);  // xy
          Kokkos::atomic_add(&d_virial(4), fz * xi);  // xz
          Kokkos::atomic_add(&d_virial(5), fz * yi);  // yz
        });
    Kokkos::fence();

    auto h_virial = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_virial);
    for (int n = 0; n < 6; n++) virial[n] += h_virial(n);
    if (use_runtime_long_range && reciprocal_forces_dev.defined()) {
      Kokkos::View<double[6], DeviceType> d_recip_virial("mfftorch::d_recip_virial");
      Kokkos::deep_copy(d_recip_virial, 0.0);
      Kokkos::View<float **, Kokkos::LayoutRight, DeviceType, Unmanaged> reciprocal_forces_v(
          reciprocal_forces_dev.data_ptr<float>(), nlocal, 3);
      Kokkos::parallel_for(
          "mfftorch::reciprocal_virial_fdotr", nlocal, KOKKOS_LAMBDA(const int i) {
            const double xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);
            const double fx = static_cast<double>(reciprocal_forces_v(i, 0));
            const double fy = static_cast<double>(reciprocal_forces_v(i, 1));
            const double fz = static_cast<double>(reciprocal_forces_v(i, 2));
            Kokkos::atomic_add(&d_recip_virial(0), fx * xi);
            Kokkos::atomic_add(&d_recip_virial(1), fy * yi);
            Kokkos::atomic_add(&d_recip_virial(2), fz * zi);
            Kokkos::atomic_add(&d_recip_virial(3), fy * xi);
            Kokkos::atomic_add(&d_recip_virial(4), fz * xi);
            Kokkos::atomic_add(&d_recip_virial(5), fz * yi);
          });
      Kokkos::fence();
      auto h_recip_virial = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_recip_virial);
      for (int n = 0; n < 6; n++) virial[n] += h_recip_virial(n);
    }
  }
#endif
  if (debug_kk_timings) {
    Kokkos::fence();
    const auto total_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_total_start).count();
    std::fprintf(
        stderr,
        "[MFF_DEBUG_KK_TIMINGS] nlocal=%d ntotal=%d nedges=%lld edge_count_ms=%.3f offset_scan_ms=%.3f fill_ms=%.3f prepare_ms=%.3f compute_ms=%.3f reciprocal_ms=%.3f force_ms=%.3f atom_output_ms=%.3f virial_ms=%.3f total_ms=%.3f\n",
        nlocal,
        ntotal,
        static_cast<long long>(Etotal),
        edge_count_ms,
        offset_scan_ms,
        fill_ms,
        prepare_ms,
        compute_ms,
        reciprocal_ms,
        force_ms,
        atom_output_ms,
        virial_ms,
        total_ms);
  }
}

namespace LAMMPS_NS {
template class PairMFFTorchKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairMFFTorchKokkos<LMPHostType>;
#endif
}  // namespace LAMMPS_NS

#endif  // LMP_KOKKOS
