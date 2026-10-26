#include "pair_mff_torch.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "input.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"
#include "variable.h"

#include "mff_periodic_table.h"
#include "mff_reciprocal_solver.h"
#include "mff_torch_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using namespace LAMMPS_NS;

namespace {

std::string normalize_variable_name(const std::string &name) {
  if (name.rfind("v_", 0) == 0) return name.substr(2);
  return name;
}

struct CellGeom {
  double cell[3][3];
  double inv[3][3];
  int pbc[3];
};

CellGeom build_cell_geom(const LAMMPS_NS::Domain *domain) {
  CellGeom g{};
  g.cell[0][0] = domain->xprd;
  g.cell[0][1] = 0.0;
  g.cell[0][2] = 0.0;
  g.cell[1][0] = domain->xy;
  g.cell[1][1] = domain->yprd;
  g.cell[1][2] = 0.0;
  g.cell[2][0] = domain->xz;
  g.cell[2][1] = domain->yz;
  g.cell[2][2] = domain->zprd;
  g.pbc[0] = domain->xperiodic;
  g.pbc[1] = domain->yperiodic;
  g.pbc[2] = domain->zperiodic;

  const double a = g.cell[0][0], b = g.cell[0][1], c = g.cell[0][2];
  const double d = g.cell[1][0], e = g.cell[1][1], f = g.cell[1][2];
  const double h = g.cell[2][0], i = g.cell[2][1], j = g.cell[2][2];
  const double det = a * (e * j - f * i) - b * (d * j - f * h) + c * (d * i - e * h);
  if (std::abs(det) < 1e-12) {
    throw std::runtime_error("mff/torch encountered a singular cell matrix");
  }
  const double inv_det = 1.0 / det;
  g.inv[0][0] = (e * j - f * i) * inv_det;
  g.inv[0][1] = (c * i - b * j) * inv_det;
  g.inv[0][2] = (b * f - c * e) * inv_det;
  g.inv[1][0] = (f * h - d * j) * inv_det;
  g.inv[1][1] = (a * j - c * h) * inv_det;
  g.inv[1][2] = (c * d - a * f) * inv_det;
  g.inv[2][0] = (d * i - e * h) * inv_det;
  g.inv[2][1] = (b * h - a * i) * inv_det;
  g.inv[2][2] = (a * e - b * d) * inv_det;
  return g;
}

inline int nearest_int(double x) {
  return (x >= 0.0) ? static_cast<int>(x + 0.5) : static_cast<int>(x - 0.5);
}

inline bool lexicographic_positive_shift(int sx, int sy, int sz) {
  return sx > 0 || (sx == 0 && sy > 0) || (sx == 0 && sy == 0 && sz > 0);
}

inline bool keep_canonical_mbd_edge(int src, int dst, int sx, int sy, int sz) {
  return src < dst || (src == dst && lexicographic_positive_shift(sx, sy, sz));
}

inline double norm3(double x, double y, double z) {
  return std::sqrt(x * x + y * y + z * z);
}

inline double periodic_face_height(const CellGeom& geom, int axis) {
  const double ax = geom.cell[0][0], ay = geom.cell[0][1], az = geom.cell[0][2];
  const double bx = geom.cell[1][0], by = geom.cell[1][1], bz = geom.cell[1][2];
  const double cx = geom.cell[2][0], cy = geom.cell[2][1], cz = geom.cell[2][2];
  const double det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
  const double volume = std::abs(det);
  if (axis == 0) return volume / std::max(norm3(by * cz - bz * cy, bz * cx - bx * cz, bx * cy - by * cx), 1.0e-12);
  if (axis == 1) return volume / std::max(norm3(cy * az - cz * ay, cz * ax - cx * az, cx * ay - cy * ax), 1.0e-12);
  return volume / std::max(norm3(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx), 1.0e-12);
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

PairMFFTorch::PairMFFTorch(LAMMPS *lmp) : Pair(lmp) {
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  comm_reverse = 3;
  comm_forward = 2;
  respa_enable = 0;
}

PairMFFTorch::~PairMFFTorch() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

int PairMFFTorch::pack_forward_comm(int n,int *list,double *buf,int,int *) {
  int offset=0;
  for(int j=0;j<n;++j)for(int k=0;k<2;++k)
    buf[offset++]=owner_map_buffer_[static_cast<size_t>(list[j])*2+k];
  return offset;
}

void PairMFFTorch::unpack_forward_comm(int n,int first,double *buf) {
  int offset=0;
  for(int i=first;i<first+n;++i)for(int k=0;k<2;++k)
    owner_map_buffer_[static_cast<size_t>(i)*2+k]=buf[offset++];
}

void PairMFFTorch::refresh_domain_owners() {
  if(!layerwise_exchange_ || comm->nprocs==1)return;
  // Empty domains cannot skip RCCL collectives while peers evaluate layers.
  // Check collectively before any per-rank early return or model call.
  int minimum=0;
  MPI_Allreduce(&atom->nlocal,&minimum,1,MPI_INT,MPI_MIN,world);
  if(minimum==0)error->all(FLERR,"Native MFF requires at least one owned atom on every MPI rank");
  const int total=atom->nlocal+atom->nghost;
  owner_map_buffer_.assign(static_cast<size_t>(total)*2,-1.0);
  for(int i=0;i<atom->nlocal;++i) {
    owner_map_buffer_[static_cast<size_t>(i)*2]=comm->me;
    owner_map_buffer_[static_cast<size_t>(i)*2+1]=i;
  }
  // Double owner codes exactly represent every LAMMPS local int index. The
  // native current communication plan handles migration and repeated images.
  comm->forward_comm(this);
  auto owners=torch::empty({total,2},torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  auto *dst=owners.data_ptr<int64_t>();
  for(size_t i=0;i<owner_map_buffer_.size();++i)dst[i]=static_cast<int64_t>(owner_map_buffer_[i]);
  engine_->set_domain_owners(owners,comm->me,comm->nprocs);
}

int PairMFFTorch::pack_reverse_comm(int n,int first,double *buf) {
  int offset=0;
  for(int i=first;i<first+n;++i)for(int axis=0;axis<3;++axis)
    buf[offset++]=reverse_force_buffer_[static_cast<size_t>(i)*3+axis];
  return offset;
}

void PairMFFTorch::unpack_reverse_comm(int n,int *list,double *buf) {
  int offset=0;
  for(int j=0;j<n;++j)for(int axis=0;axis<3;++axis) {
    const int i=list[j];const double value=buf[offset++];
    reverse_force_buffer_[static_cast<size_t>(i)*3+axis]+=value;
    // Intermediate ghosts keep their accumulated contribution for the next hop.
  }
}

void PairMFFTorch::reverse_ghost_contributions(const torch::Tensor& forces,int ntotal,bool add_to_host) {
  reverse_force_buffer_.assign(static_cast<size_t>(ntotal)*3,0.0);
  if(forces.defined()) {
    auto cpu=forces.to(torch::kCPU,torch::kFloat64).contiguous();
    if(cpu.numel()!=static_cast<int64_t>(ntotal)*3)throw std::runtime_error("Invalid ghost force buffer shape");
    const double *data=cpu.data_ptr<double>();
    std::copy(data,data+cpu.numel(),reverse_force_buffer_.begin());
  }
  // Local model forces have already been written. Return only received ghost
  // derivatives, independently of AtomKokkos's host/device force sync state.
  std::fill(reverse_force_buffer_.begin(),reverse_force_buffer_.begin()+static_cast<size_t>(atom->nlocal)*3,0.0);
  comm->reverse_comm(this);
  if(add_to_host)for(int i=0;i<atom->nlocal;++i)for(int axis=0;axis<3;++axis)
    atom->f[i][axis]+=reverse_force_buffer_[static_cast<size_t>(i)*3+axis];
}

void PairMFFTorch::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= n; j++) {
      setflag[i][j] = 0;
      cutsq[i][j] = 0.0;
    }
  }
}

void PairMFFTorch::configure_message_depth() {
  int exported_depth=0;
  layerwise_exchange_=false;
  std::ifstream metadata(core_pt_path_+".meta");
  std::string key;
  while(metadata>>key) {
    if(key=="message_passing_depth")metadata>>exported_depth;
    else if(key=="feature_exchange") {
      std::string transport;metadata>>transport;
      layerwise_exchange_=(transport=="rccl");
    }
    else {std::string rest;std::getline(metadata,rest);}
  }
  if(exported_depth>0) {
    if(mp_depth_explicit_ && mp_depth_!=exported_depth)
      error->all(FLERR,"mff/torch depth disagrees with the exported message_passing_depth");
    mp_depth_=exported_depth;
  }
}

void PairMFFTorch::settings(int narg, char **arg) {
  if (narg < 1) error->all(FLERR, "Illegal pair_style mff/torch command");
  cut_global_ = utils::numeric(FLERR, arg[0], false, lmp);
  if (cut_global_ <= 0.0) error->all(FLERR, "pair_style mff/torch cutoff must be > 0");
  cutsq_global_ = cut_global_ * cut_global_;
  dispersion_cut_global_ = 0.0;
  dispersion_cutsq_global_ = 0.0;
  request_cut_global_ = cut_global_;
  request_cutsq_global_ = cutsq_global_;
  mp_depth_=2;
  mp_depth_explicit_=false;

  use_external_field_ = false;
  use_electric_field_ = false;
  use_magnetic_field_ = false;
  use_rank2_external_field_ = false;
  external_field_symmetric_rank2_ = false;
  use_fidelity_input_ = false;
  fidelity_is_variable_ = false;
  fidelity_var_name_.clear();
  fidelity_constant_ = 0;
  electric_field_var_names_.clear();
  magnetic_field_var_names_.clear();
  rank2_external_field_var_names_.clear();
  cached_external_field_values_.clear();
  external_tensor_cache_ = torch::Tensor();

  for (int i = 1; i < narg; ++i) {
    const std::string opt(arg[i]);
    if (opt == "cpu" || opt == "cuda") {
      device_str_ = opt;
      continue;
    }
    if(opt=="depth") {
      if(i+1>=narg)error->all(FLERR,"mff/torch depth expects a positive integer");
      mp_depth_=utils::inumeric(FLERR,arg[++i],false,lmp);
      if(mp_depth_<1)error->all(FLERR,"mff/torch depth must be positive");
      mp_depth_explicit_=true;
      continue;
    }
    if (opt == "dispersion") {
      error->all(FLERR, "Dispersion is not supported by this K100 distribution");
    }
    if (opt == "field") {
      if (i + 3 >= narg) {
        error->all(FLERR, "pair_style mff/torch field expects three equal-style variables: v_Ex v_Ey v_Ez");
      }
      use_external_field_ = true;
      use_electric_field_ = true;
      electric_field_var_names_ = {
          normalize_variable_name(arg[i + 1]),
          normalize_variable_name(arg[i + 2]),
          normalize_variable_name(arg[i + 3]),
      };
      i += 3;
      continue;
    }
    if (opt == "mfield") {
      if (i + 3 >= narg) {
        error->all(FLERR, "pair_style mff/torch mfield expects three equal-style variables: v_Bx v_By v_Bz");
      }
      use_external_field_ = true;
      use_magnetic_field_ = true;
      magnetic_field_var_names_ = {
          normalize_variable_name(arg[i + 1]),
          normalize_variable_name(arg[i + 2]),
          normalize_variable_name(arg[i + 3]),
      };
      i += 3;
      continue;
    }
    if (opt == "field9") {
      if (i + 9 >= narg) {
        error->all(FLERR,
                   "pair_style mff/torch field9 expects nine equal-style variables "
                   "(row-major: xx xy xz yx yy yz zx zy zz)");
      }
      use_external_field_ = true;
      use_rank2_external_field_ = true;
      external_field_symmetric_rank2_ = false;
      rank2_external_field_var_names_.clear();
      for (int k = 1; k <= 9; ++k) rank2_external_field_var_names_.push_back(normalize_variable_name(arg[i + k]));
      i += 9;
      continue;
    }
    if (opt == "field6") {
      if (i + 6 >= narg) {
        error->all(FLERR,
                   "pair_style mff/torch field6 expects six equal-style variables "
                   "(symmetric order: xx yy zz xy xz yz)");
      }
      use_external_field_ = true;
      use_rank2_external_field_ = true;
      external_field_symmetric_rank2_ = true;
      rank2_external_field_var_names_.clear();
      for (int k = 1; k <= 6; ++k) rank2_external_field_var_names_.push_back(normalize_variable_name(arg[i + k]));
      i += 6;
      continue;
    }
    if (opt == "fidelity") {
      if (i + 1 >= narg) {
        error->all(FLERR, "pair_style mff/torch fidelity expects an integer or equal-style variable");
      }
      use_fidelity_input_ = true;
      std::string value(arg[i + 1]);
      if (value.rfind("v_", 0) == 0) {
        fidelity_is_variable_ = true;
        fidelity_var_name_ = normalize_variable_name(value);
      } else {
        fidelity_is_variable_ = false;
        fidelity_constant_ = static_cast<int64_t>(std::stoll(value));
      }
      i += 1;
      continue;
    }
    error->all(FLERR, ("Unknown pair_style mff/torch option: " + opt).c_str());
  }

  if (use_rank2_external_field_ && (use_electric_field_ || use_magnetic_field_)) {
    error->all(FLERR, "pair_style mff/torch does not yet support combining field6/field9 with field/mfield");
  }
  if (use_electric_field_ && use_magnetic_field_) {
    cached_external_field_values_.assign(6, 0.0f);
  } else if (use_rank2_external_field_) {
    cached_external_field_values_.assign(external_field_symmetric_rank2_ ? 6 : 9, 0.0f);
  } else if (use_electric_field_ || use_magnetic_field_) {
    cached_external_field_values_.assign(3, 0.0f);
  }
}

void PairMFFTorch::coeff(int narg, char **arg) {
  if (!allocated) allocate();
  if (narg < 3) error->all(FLERR, "Illegal pair_coeff command for mff/torch");

  // Expect: pair_coeff * * core.pt <elem1> <elem2> ... (ntypes entries)
  // arg[0], arg[1] are * *
  core_pt_path_ = std::string(arg[2]);

  const int ntypes = atom->ntypes;
  if (narg != 3 + ntypes) error->all(FLERR, "pair_coeff mff/torch expects one element symbol per atom type");

  type2Z_.assign(ntypes + 1, 0);
  for (int itype = 1; itype <= ntypes; itype++) {
    const std::string sym(arg[2 + itype]);
    if (sym == "NULL" || sym == "null") {
      type2Z_[itype] = 0;
      continue;
    }
    int Z = mfftorch::symbol_to_Z(sym);
    if (Z <= 0) error->all(FLERR, ("Unknown element symbol in pair_coeff mff/torch: " + sym).c_str());
    type2Z_[itype] = static_cast<int64_t>(Z);
  }

  for (int i = 1; i <= ntypes; i++) {
    for (int j = i; j <= ntypes; j++) {
      setflag[i][j] = 1;
      cutsq[i][j] = request_cutsq_global_;
      setflag[j][i] = 1;
      cutsq[j][i] = request_cutsq_global_;
    }
  }

  if (!engine_) engine_ = std::make_unique<mfftorch::MFFTorchEngine>();
  engine_loaded_ = false;  // lazy load at init_style/compute
}

void PairMFFTorch::init_style() {
  if (core_pt_path_.empty()) error->all(FLERR, "pair_coeff for mff/torch must specify core.pt path");
  configure_message_depth();

  // A num_interaction-layer message-passing model needs each LOCAL atom's full K-hop environment.
  // Two ways to supply it without per-layer communication:
  //   * SINGLE RANK (nprocs==1, fold_mode_): fold each periodic-ghost neighbour back to its LOCAL
  //     owner (atom->map) + an integer cell shift, so the model graph is just the nlocal local nodes
  //     with PBC-shift edges (the exact training topology). A 1x-cutoff ghost halo is enough (we only
  //     need to SEE the neighbour to fold it), and every ghost's owner is local so its features are
  //     complete. This makes the model run on nlocal instead of the mp_depth_*cutoff halo (~12x fewer
  //     nodes on a small periodic cell) -> ~12x less memory/compute, so much larger systems fit.
  //   * MULTI RANK (nprocs>1, refined-A): folding breaks because a boundary ghost's owner is on
  //     another rank. Instead request a GHOST neighbor list so ghosts are also graph centers, and
  //     extend the halo to mp_depth_*cutoff so every local atom's K-hop neighbours are present as
  //     ghosts. Correct under MPI domain decomposition; keep only the LOCAL energies/forces.
  fold_mode_ = (comm->nprocs == 1);
  if (fold_mode_) {
    if (atom->map_style == Atom::MAP_NONE)
      error->all(FLERR, "pair_style mff/torch (single-rank) needs an atom map to fold periodic ghosts "
                        "to local atoms; add 'atom_modify map yes' (or array/hash) to your input.");
    neighbor->add_request(this, NeighConst::REQ_FULL);  // 1x halo: default ghost cutoff is enough
  } else {
    neighbor->add_request(this, layerwise_exchange_ ? NeighConst::REQ_FULL :
                          NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
    const double halo = std::max(static_cast<double>(layerwise_exchange_ ? 1 : mp_depth_) * cut_global_, request_cut_global_);
    if (comm->cutghostuser < halo) comm->cutghostuser = halo;
  }

  try {
    if (!engine_) engine_ = std::make_unique<mfftorch::MFFTorchEngine>();
    if (!reciprocal_solver_) reciprocal_solver_ = std::make_unique<mfftorch::MFFReciprocalSolver>();
    engine_->load_core(core_pt_path_, device_str_);
    if (reciprocal_solver_) {
      auto cfg = reciprocal_solver_->config();
      // Use the exported (training) mesh size so the deployed reciprocal grid matches training.
      // Previously the engine never read long_range_mesh_size from the export, so the solver was
      // stuck at its default/env mesh (16) regardless of the trained grid.
      cfg.mesh_size = static_cast<int>(engine_->long_range_mesh_size());
      cfg.max_multipole_l = static_cast<int>(engine_->long_range_max_multipole_l());
      cfg.source_channels = static_cast<int>(engine_->reciprocal_source_channels());
      cfg.slab_padding_factor = static_cast<int>(engine_->reciprocal_source_slab_padding_factor());
      cfg.green_mode = (engine_->long_range_green_mode() == "learned_poisson")
                           ? mfftorch::ReciprocalGreenMode::LearnedPoisson
                           : mfftorch::ReciprocalGreenMode::Poisson;
      // Latent-multipole alignment with the in-model MeshLongRangeKernel3D.multipole_energy:
      // Ewald Gaussian screening (+ its alpha prefactor) and the learned reciprocal energy scale.
      cfg.full_ewald = engine_->long_range_mesh_fft_full_ewald();
      cfg.ewald_alpha_prefactor = engine_->long_range_ewald_alpha_prefactor();
      cfg.energy_scale = engine_->long_range_energy_scale();
      // Mesh assignment (pcs vs cic) must match the in-model PME or the multipole reciprocal
      // energy/forces drift by a few % at coarse meshes; the C++ cuFFT solver supports both.
      cfg.assignment = (engine_->long_range_assignment() == "pcs") ? 1 : 0;
      reciprocal_solver_->set_config(cfg);
    }

    validate_external_field_configuration();
    engine_loaded_ = true;
    engine_->warmup(32, 256);
  } catch (const std::exception &e) {
    error->all(FLERR, (std::string("Failed to load TorchScript core: ") + e.what()).c_str());
  }
}

double PairMFFTorch::init_one(int i, int j) {
  return request_cut_global_;
}

void PairMFFTorch::validate_external_field_configuration() {
  if (!engine_) return;

  const double model_dispersion_cutoff = engine_->dispersion_cutoff();
  if (engine_->requires_mbd_dispersion_edges()) {
    if (dispersion_cut_global_ <= 0.0) {
      error->all(FLERR,
                 "core.pt/.pt2 was trained/exported with MBD dispersion_cutoff > 0, but pair_style "
                 "mff/torch was not given a dispersion neighbor list. Add 'dispersion <cutoff>' "
                 "with the checkpoint dispersion_cutoff.");
    }
    const double tol = 1.0e-6 * std::max(1.0, std::abs(model_dispersion_cutoff));
    if (std::abs(dispersion_cut_global_ - model_dispersion_cutoff) > tol) {
      error->all(
          FLERR,
          ("pair_style mff/torch dispersion cutoff (" + std::to_string(dispersion_cut_global_) +
           ") does not match the model MBD dispersion_cutoff (" + std::to_string(model_dispersion_cutoff) +
           "); use the trained cutoff so deployment and training build the same dispersion graph rule.")
              .c_str());
    }
  }

  if (use_external_field_) {
    if (!engine_->accepts_external_tensor()) {
      error->all(FLERR,
                 "pair_style mff/torch field/mfield was specified, but core.pt does not accept external_tensor");
    }
    const bool expects_field_1o = engine_->external_tensor_has_field_1o() || engine_->external_tensor_irrep() == "1o";
    const bool expects_field_1e = engine_->external_tensor_has_field_1e() || engine_->external_tensor_irrep() == "1e";
    if (use_rank2_external_field_) {
      if (expects_field_1o || expects_field_1e) {
        error->all(FLERR, "core.pt expects rank-1 external fields; use field and/or mfield instead of field6/field9");
      }
      const int expected_nvars = external_field_symmetric_rank2_ ? 6 : 9;
      if (static_cast<int>(rank2_external_field_var_names_.size()) != expected_nvars) {
        error->all(FLERR, "mff/torch external field variable count does not match the selected field mode");
      }
    } else {
      if (expects_field_1o && !expects_field_1e && !use_electric_field_) {
        error->all(FLERR, "core.pt expects electric-field-style external_tensor; use pair_style mff/torch ... field v_Ex v_Ey v_Ez");
      }
      if (expects_field_1e && !expects_field_1o && !use_magnetic_field_) {
        error->all(FLERR, "core.pt expects magnetic-field-style external_tensor; use pair_style mff/torch ... mfield v_Bx v_By v_Bz");
      }
      if (use_electric_field_ && !expects_field_1o && expects_field_1e) {
        error->all(FLERR, "core.pt expects magnetic-field-style external_tensor only; remove field and use mfield");
      }
      if (use_magnetic_field_ && !expects_field_1e && expects_field_1o) {
        error->all(FLERR, "core.pt expects electric-field-style external_tensor only; remove mfield and use field");
      }
    }

    auto validate_names = [&](const std::vector<std::string>& names) {
      for (const auto& name : names) {
        if (name.empty()) error->all(FLERR, "pair_style mff/torch external field variable name is empty");
        const int ivar = input->variable->find(name.c_str());
        if (ivar < 0) {
          error->all(FLERR, ("Unknown LAMMPS variable for mff/torch field: " + name).c_str());
        }
        if (!input->variable->equalstyle(ivar)) {
          error->all(FLERR, ("mff/torch field variables must be equal-style scalars: " + name).c_str());
        }
      }
    };
    validate_names(electric_field_var_names_);
    validate_names(magnetic_field_var_names_);
    validate_names(rank2_external_field_var_names_);
  } else if (engine_->accepts_external_tensor()) {
    error->all(FLERR,
               "core.pt requires external_tensor, but pair_style mff/torch was not given field/mfield/field6/field9");
  }

  if (engine_->requires_runtime_fidelity()) {
    if (!use_fidelity_input_) {
      error->all(FLERR, "core.pt requires runtime fidelity_ids; add pair_style mff/torch ... fidelity <int|v_name>");
    }
  } else if (use_fidelity_input_ && !engine_->takes_fidelity_arg()) {
    error->all(FLERR, "pair_style mff/torch fidelity was specified, but core.pt does not accept fidelity_ids");
  }
  if (use_fidelity_input_ && fidelity_is_variable_) {
    if (fidelity_var_name_.empty()) error->all(FLERR, "pair_style mff/torch fidelity variable name is empty");
    const int ivar = input->variable->find(fidelity_var_name_.c_str());
    if (ivar < 0) {
      error->all(FLERR, ("Unknown LAMMPS variable for mff/torch fidelity: " + fidelity_var_name_).c_str());
    }
    if (!input->variable->equalstyle(ivar)) {
      error->all(FLERR, ("mff/torch fidelity variable must be an equal-style scalar: " + fidelity_var_name_).c_str());
    }
  }
  if (use_fidelity_input_ && !fidelity_is_variable_ && engine_->num_fidelity_levels() > 0) {
    if (fidelity_constant_ < 0 || fidelity_constant_ >= engine_->num_fidelity_levels()) {
      error->all(
          FLERR,
          ("pair_style mff/torch fidelity is out of range [0, " + std::to_string(engine_->num_fidelity_levels() - 1) +
           "]: " + std::to_string(fidelity_constant_))
              .c_str());
    }
  }
}

torch::Tensor PairMFFTorch::current_external_tensor(const torch::Device& device) {
  if (!use_external_field_) return torch::Tensor();

  std::vector<float> values;
  auto append_values = [&](const std::vector<std::string>& names) {
    for (const auto& name : names) {
      const int ivar = input->variable->find(name.c_str());
      if (ivar < 0) {
        error->all(FLERR, ("Unknown LAMMPS variable for mff/torch field: " + name).c_str());
      }
      values.push_back(static_cast<float>(input->variable->compute_equal(ivar)));
    }
  };
  if (use_rank2_external_field_) {
    append_values(rank2_external_field_var_names_);
  } else {
    if (use_electric_field_) append_values(electric_field_var_names_);
    if (use_magnetic_field_) append_values(magnetic_field_var_names_);
  }

  const bool cache_hit =
      external_tensor_cache_.defined() &&
      external_tensor_cache_.device() == device &&
      cached_external_field_values_ == values;
  if (cache_hit) return external_tensor_cache_;

  cached_external_field_values_ = values;
  torch::Tensor cpu;
  if (!use_rank2_external_field_) {
    cpu = torch::tensor(values, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  } else if (external_field_symmetric_rank2_) {
    cpu = torch::tensor(
              {
                  values[0], values[3], values[4],
                  values[3], values[1], values[5],
                  values[4], values[5], values[2],
              },
              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
              .reshape({3, 3});
  } else {
    cpu = torch::tensor(values, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
              .reshape({3, 3});
  }
  external_tensor_cache_ = (device.is_cpu()) ? cpu : cpu.to(device);
  return external_tensor_cache_;
}

torch::Tensor PairMFFTorch::current_fidelity_tensor(const torch::Device& device) {
  if (!use_fidelity_input_) return torch::Tensor();

  int64_t fidelity_value = fidelity_constant_;
  if (fidelity_is_variable_) {
    const int ivar = input->variable->find(fidelity_var_name_.c_str());
    if (ivar < 0) {
      error->all(FLERR, ("Unknown LAMMPS variable for mff/torch fidelity: " + fidelity_var_name_).c_str());
    }
    fidelity_value = static_cast<int64_t>(std::llround(input->variable->compute_equal(ivar)));
  }
  if (engine_ && engine_->num_fidelity_levels() > 0) {
    if (fidelity_value < 0 || fidelity_value >= engine_->num_fidelity_levels()) {
      error->all(
          FLERR,
          ("mff/torch fidelity value is out of range [0, " + std::to_string(engine_->num_fidelity_levels() - 1) +
           "]: " + std::to_string(fidelity_value))
              .c_str());
    }
  }
  auto cpu = torch::tensor({fidelity_value}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  return device.is_cpu() ? cpu : cpu.to(device);
}

void PairMFFTorch::reset_physical_outputs() {
  global_phys_cpu_ = torch::Tensor();
  atom_phys_cpu_ = torch::Tensor();
  global_phys_mask_cpu_ = torch::Tensor();
  atom_phys_mask_cpu_ = torch::Tensor();
  cached_phys_timestep_ = update ? static_cast<int64_t>(update->ntimestep) : -1;
}

void PairMFFTorch::cache_physical_outputs(const mfftorch::MFFOutputs& out, int nlocal) {
  cached_phys_timestep_ = update ? static_cast<int64_t>(update->ntimestep) : -1;
  if (!physical_cache_requested_) {
    global_phys_cpu_ = torch::Tensor();
    atom_phys_cpu_ = torch::Tensor();
    global_phys_mask_cpu_ = torch::Tensor();
    atom_phys_mask_cpu_ = torch::Tensor();
    return;
  }

  if (out.global_phys.defined()) {
    global_phys_cpu_ = out.global_phys.to(torch::kCPU, torch::kFloat64).contiguous();
  } else {
    global_phys_cpu_ = torch::Tensor();
  }
  if (out.atom_phys.defined()) {
    auto atom_phys = out.atom_phys.to(torch::kCPU, torch::kFloat64).contiguous();
    if (atom_phys.dim() >= 2 && atom_phys.size(0) >= nlocal) {
      atom_phys_cpu_ = atom_phys.narrow(0, 0, nlocal).clone();
    } else {
      atom_phys_cpu_ = atom_phys.clone();
    }
  } else {
    atom_phys_cpu_ = torch::Tensor();
  }
  global_phys_mask_cpu_ = out.global_phys_mask.defined()
                              ? out.global_phys_mask.to(torch::kCPU, torch::kFloat64).contiguous()
                              : torch::Tensor();
  atom_phys_mask_cpu_ = out.atom_phys_mask.defined()
                            ? out.atom_phys_mask.to(torch::kCPU, torch::kFloat64).contiguous()
                            : torch::Tensor();
}

void PairMFFTorch::compute(int eflag, int vflag) {
  const char *profile_env = std::getenv("MFF_PROFILE_PAIR");
  const bool profile_pair = profile_env && profile_env[0] && profile_env[0] != '0';
  using ProfileClock = std::chrono::steady_clock;
  const auto profile_start = profile_pair ? ProfileClock::now() : ProfileClock::time_point{};
  ev_init(eflag, vflag);
  reset_physical_outputs();

  if (!engine_loaded_) init_style();

  const int nlocal = atom->nlocal;
  const int nghost = atom->nghost;
  const int ntotal = nlocal + nghost;
  refresh_domain_owners();
  // Number of NODES the model sees: fold_mode_ collapses every ghost onto its local owner, so the
  // graph is just the nlocal local atoms; refined-A keeps the ghosts as nodes (ntotal).
  const int n_model = fold_mode_ ? nlocal : ntotal;

  // Neighbor list
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  tagint *tag = atom->tag;

  std::unordered_map<tagint, int> local_owner_by_tag;
  if (fold_mode_) {
    local_owner_by_tag.reserve(static_cast<size_t>(std::max(nlocal, 0)));
    for (int i = 0; i < nlocal; ++i) {
      local_owner_by_tag[tag[i]] = i;
    }
  }

  // Build type->Z mapped A (CPU then move to engine device).
  // Reuse persistent buffers to avoid heap allocation every step.
  buf_A_cpu_.resize(static_cast<size_t>(n_model));
  buf_pos_cpu_.resize(static_cast<size_t>(n_model) * 3);
  for (int i = 0; i < n_model; i++) {
    const int itype = type[i];
    buf_A_cpu_[i] = (itype >= 0 && itype < static_cast<int>(type2Z_.size())) ? type2Z_[itype] : 0;
    buf_pos_cpu_[static_cast<size_t>(i) * 3 + 0] = static_cast<float>(x[i][0]);
    buf_pos_cpu_[static_cast<size_t>(i) * 3 + 1] = static_cast<float>(x[i][1]);
    buf_pos_cpu_[static_cast<size_t>(i) * 3 + 2] = static_cast<float>(x[i][2]);
  }

  const CellGeom geom = build_cell_geom(domain);
  if (engine_->requires_mbd_dispersion_edges()) {
    validate_mbd_dispersion_single_image_cutoff(error, geom, dispersion_cut_global_, "pair_style mff/torch");
  }
  float cell_cpu[9] = {
      static_cast<float>(geom.cell[0][0]), static_cast<float>(geom.cell[0][1]), static_cast<float>(geom.cell[0][2]),
      static_cast<float>(geom.cell[1][0]), static_cast<float>(geom.cell[1][1]), static_cast<float>(geom.cell[1][2]),
      static_cast<float>(geom.cell[2][0]), static_cast<float>(geom.cell[2][1]), static_cast<float>(geom.cell[2][2]),
  };
  auto cell_t = torch::from_blob(cell_cpu, {1, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
  if (nlocal == 0) {
    const bool exports_runtime_source = engine_->exports_reciprocal_source() && engine_->reciprocal_source_channels() > 0;

    const bool use_reciprocal =
        reciprocal_solver_ && exports_runtime_source;
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
            cell_t,
            geom,
            static_cast<bool>(eflag),
            torch::Device(torch::kCPU));

          (void)reciprocal_solver_->compute(reciprocal_inputs);

      } catch (const std::exception &e) {
        error->all(FLERR, (std::string("mff/torch runtime long-range solver failed on empty rank: ") + e.what()).c_str());
      }
    }
    if(comm->nprocs>1 && !force->newton_pair)reverse_ghost_contributions(torch::Tensor(),ntotal);
    return;
  }

  // Count edges (upper bound) and build edges + lattice shifts (reuse persistent buffers).
  const bool use_dispersion_edges = dispersion_cut_global_ > 0.0;
  if (fold_mode_) {
    // FOLD (single rank): edges from every local center i to its neighbours, each neighbour folded to
    // its LOCAL owner jl + integer cell offset g (x[j]=x[jl]+g@cell). Edge (src=jl, dst=i, shift=-g) ->
    // edge_vec = x[i]-x[jl]-g@cell = x[i]-x[j] (correct), and the model aggregates the neighbour's
    // features INTO local center i. All indices are < nlocal, so the model runs on the nlocal local
    // nodes only (no ghost nodes). Validated to give PotEng == DFT; forces land on local owners.
    int64_t Emax = 0;
    for (int ii = 0; ii < inum; ii++) Emax += numneigh[ilist[ii]];
    buf_edge_src_cpu_.clear();
    buf_edge_dst_cpu_.clear();
    buf_edge_shifts_cpu_.clear();
    buf_disp_edge_src_cpu_.clear();
    buf_disp_edge_dst_cpu_.clear();
    buf_disp_edge_shifts_cpu_.clear();
    buf_edge_src_cpu_.reserve(static_cast<size_t>(Emax));
    buf_edge_dst_cpu_.reserve(static_cast<size_t>(Emax));
    buf_edge_shifts_cpu_.reserve(static_cast<size_t>(Emax) * 3);
    if (use_dispersion_edges) {
      buf_disp_edge_src_cpu_.reserve(static_cast<size_t>(Emax));
      buf_disp_edge_dst_cpu_.reserve(static_cast<size_t>(Emax));
      buf_disp_edge_shifts_cpu_.reserve(static_cast<size_t>(Emax) * 3);
    }
    for (int ii = 0; ii < inum; ii++) {
      int i = ilist[ii];
      int jnum = numneigh[i];
      int *jlist = firstneigh[i];
      for (int jj = 0; jj < jnum; jj++) {
        int j = jlist[jj] & NEIGHMASK;
        const double rawx = x[j][0] - x[i][0];
        const double rawy = x[j][1] - x[i][1];
        const double rawz = x[j][2] - x[i][2];
        const double fracx = rawx * geom.inv[0][0] + rawy * geom.inv[1][0] + rawz * geom.inv[2][0];
        const double fracy = rawx * geom.inv[0][1] + rawy * geom.inv[1][1] + rawz * geom.inv[2][1];
        const double fracz = rawx * geom.inv[0][2] + rawy * geom.inv[1][2] + rawz * geom.inv[2][2];
        // The neighbor's ghost position already supplies its periodic image.
        const int sx = 0;
        const int sy = 0;
        const int sz = 0;
        const double delx = rawx + sx * geom.cell[0][0] + sy * geom.cell[1][0] + sz * geom.cell[2][0];
        const double dely = rawy + sx * geom.cell[0][1] + sy * geom.cell[1][1] + sz * geom.cell[2][1];
        const double delz = rawz + sx * geom.cell[0][2] + sy * geom.cell[1][2] + sz * geom.cell[2][2];
        const double rsq = delx * delx + dely * dely + delz * delz;
        if (rsq > cutsq_global_ && (!use_dispersion_edges || rsq > dispersion_cutsq_global_)) continue;
        const auto owner_it = local_owner_by_tag.find(tag[j]);
        if (owner_it == local_owner_by_tag.end()) continue;
        const int jl = owner_it->second;
        const double dxl = x[j][0] - x[jl][0];
        const double dyl = x[j][1] - x[jl][1];
        const double dzl = x[j][2] - x[jl][2];
        const int gx = nearest_int(dxl * geom.inv[0][0] + dyl * geom.inv[1][0] + dzl * geom.inv[2][0]);
        const int gy = nearest_int(dxl * geom.inv[0][1] + dyl * geom.inv[1][1] + dzl * geom.inv[2][1]);
        const int gz = nearest_int(dxl * geom.inv[0][2] + dyl * geom.inv[1][2] + dzl * geom.inv[2][2]);
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
        // (src, dst, shift) ~ (dst, src, -shift).  Self-image couplings keep the
        // lexicographically positive shift half, matching the Python builder.
        if (use_dispersion_edges && rsq <= dispersion_cutsq_global_ &&
            keep_canonical_mbd_edge(jl, i, out_sx, out_sy, out_sz)) {
          buf_disp_edge_src_cpu_.push_back(static_cast<int64_t>(jl));
          buf_disp_edge_dst_cpu_.push_back(static_cast<int64_t>(i));
          buf_disp_edge_shifts_cpu_.push_back(static_cast<float>(out_sx));
          buf_disp_edge_shifts_cpu_.push_back(static_cast<float>(out_sy));
          buf_disp_edge_shifts_cpu_.push_back(static_cast<float>(out_sz));
        }
      }
    }
  } else {
  // --- Pick CENTERS: local atoms + ghosts within (mp_depth_-1) hops of a local atom. Only centers
  // get incoming edges (a center must AGGREGATE to produce a correct deeper-layer feature); ghosts
  // beyond that are SRC-only NODES -- their layer-0 embedding is all a center needs from them. This
  // keeps each local atom's full K-hop environment exact while avoiding the edge blow-up (and OOM)
  // of making EVERY 2x-cutoff halo ghost a center. (REQ_GHOST: ilist is inum local then gnum ghost.)
  std::vector<char> is_center(static_cast<size_t>(ntotal), 0);
  std::vector<int> frontier;
  frontier.reserve(static_cast<size_t>(inum));
  for (int ii = 0; ii < inum; ii++) { is_center[ilist[ii]] = 1; frontier.push_back(ilist[ii]); }
  for (int hop = 1; hop < (layerwise_exchange_ ? 1 : mp_depth_); hop++) {
    std::vector<int> next;
    for (int ci : frontier) {
      const int jn = numneigh[ci];
      const int *jl = firstneigh[ci];
      for (int jj = 0; jj < jn; jj++) {
        const int j = jl[jj] & NEIGHMASK;
        const double dx = x[j][0] - x[ci][0], dy = x[j][1] - x[ci][1], dz = x[j][2] - x[ci][2];
        if (dx * dx + dy * dy + dz * dz > cutsq_global_) continue;  // ghosts carry the image -> direct dist
        if (!is_center[j]) { is_center[j] = 1; next.push_back(j); }
      }
    }
    frontier.swap(next);
  }

  // Count edges (upper bound), ONLY for centers (non-center ghosts stay src-only nodes).
  const int ncenters = inum + list->gnum;
  int64_t Emax = 0;
  for (int ii = 0; ii < ncenters; ii++) {
    const int i = ilist[ii];
    if (is_center[i]) Emax += numneigh[i];
  }
  buf_edge_src_cpu_.clear();
  buf_edge_dst_cpu_.clear();
  buf_edge_shifts_cpu_.clear();
  buf_disp_edge_src_cpu_.clear();
  buf_disp_edge_dst_cpu_.clear();
  buf_disp_edge_shifts_cpu_.clear();
  buf_edge_src_cpu_.reserve(static_cast<size_t>(Emax));
  buf_edge_dst_cpu_.reserve(static_cast<size_t>(Emax));
  buf_edge_shifts_cpu_.reserve(static_cast<size_t>(Emax) * 3);
  if (use_dispersion_edges) {
    buf_disp_edge_src_cpu_.reserve(static_cast<size_t>(Emax));
    buf_disp_edge_dst_cpu_.reserve(static_cast<size_t>(Emax));
    buf_disp_edge_shifts_cpu_.reserve(static_cast<size_t>(Emax) * 3);
  }

  for (int ii = 0; ii < ncenters; ii++) {
    int i = ilist[ii];
    if (!is_center[i]) continue;   // only centers get incoming edges; non-center ghosts are src-only
    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      const double rawx = x[j][0] - x[i][0];
      const double rawy = x[j][1] - x[i][1];
      const double rawz = x[j][2] - x[i][2];
      const double fracx = rawx * geom.inv[0][0] + rawy * geom.inv[1][0] + rawz * geom.inv[2][0];
      const double fracy = rawx * geom.inv[0][1] + rawy * geom.inv[1][1] + rawz * geom.inv[2][1];
      const double fracz = rawx * geom.inv[0][2] + rawy * geom.inv[1][2] + rawz * geom.inv[2][2];
      const int sx = 0;
      const int sy = 0;
      const int sz = 0;
      const double shiftx = sx * geom.cell[0][0] + sy * geom.cell[1][0] + sz * geom.cell[2][0];
      const double shifty = sx * geom.cell[0][1] + sy * geom.cell[1][1] + sz * geom.cell[2][1];
      const double shiftz = sx * geom.cell[0][2] + sy * geom.cell[1][2] + sz * geom.cell[2][2];
      const double delx = rawx + shiftx;
      const double dely = rawy + shifty;
      const double delz = rawz + shiftz;
      const double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq > cutsq_global_ && (!use_dispersion_edges || rsq > dispersion_cutsq_global_)) continue;

      // Edge: neighbor j -> center i (the model puts edge features on edge_src and scatters into
      // edge_dst, so the CENTER must be the dst). j keeps its (possibly ghost) index -- ghosts are
      // real graph nodes here, with their own edges, so their features are correct. edge_vec =
      // x[i]-x[j]+shift@cell; the ghost already carries the periodic image (min-image sx==0 in the
      // halo), so shift = -(sx,sy,sz) is the robust value (0 for ghost edges). This is the convention
      // the model was trained with (center aggregates neighbours) and is correct under MPI domain
      // decomposition because j's features come from j-as-a-center, not from a cross-rank owner.
      if (rsq <= cutsq_global_) {
        buf_edge_src_cpu_.push_back(static_cast<int64_t>(j));
        buf_edge_dst_cpu_.push_back(static_cast<int64_t>(i));
        buf_edge_shifts_cpu_.push_back(static_cast<float>(-sx));
        buf_edge_shifts_cpu_.push_back(static_cast<float>(-sy));
        buf_edge_shifts_cpu_.push_back(static_cast<float>(-sz));
      }
      // (src, dst, shift) ~ (dst, src, -shift).  Self-image couplings keep the
      // lexicographically positive shift half, matching the Python builder.
      if (use_dispersion_edges && rsq <= dispersion_cutsq_global_ &&
          keep_canonical_mbd_edge(j, i, -sx, -sy, -sz)) {
        buf_disp_edge_src_cpu_.push_back(static_cast<int64_t>(j));
        buf_disp_edge_dst_cpu_.push_back(static_cast<int64_t>(i));
        buf_disp_edge_shifts_cpu_.push_back(static_cast<float>(-sx));
        buf_disp_edge_shifts_cpu_.push_back(static_cast<float>(-sy));
        buf_disp_edge_shifts_cpu_.push_back(static_cast<float>(-sz));
      }
    }
  }
  }  // end else (refined-A multi-rank path)

  const int64_t E = static_cast<int64_t>(buf_edge_src_cpu_.size());
  const int64_t Edisp = use_dispersion_edges ? static_cast<int64_t>(buf_disp_edge_src_cpu_.size()) : 0;
  if (!layerwise_exchange_ && E == 0 && (!use_dispersion_edges || Edisp == 0)) return;
  const auto profile_graph = profile_pair ? ProfileClock::now() : ProfileClock::time_point{};
  sort_edge_vectors(buf_edge_src_cpu_, buf_edge_dst_cpu_, buf_edge_shifts_cpu_);
  if (use_dispersion_edges) {
    sort_edge_vectors(buf_disp_edge_src_cpu_, buf_disp_edge_dst_cpu_, buf_disp_edge_shifts_cpu_);
  }
  const auto profile_sort = profile_pair ? ProfileClock::now() : ProfileClock::time_point{};
  auto validate_edge_bounds = [&](const std::vector<int64_t>& src, const std::vector<int64_t>& dst,
                                  const char* label) {
    for (size_t k = 0; k < src.size(); ++k) {
      if (src[k] < 0 || src[k] >= n_model || dst[k] < 0 || dst[k] >= n_model) {
        error->all(FLERR,
                   (std::string("mff/torch built out-of-range ") + label + " edge at " +
                    std::to_string(k) + ": src=" + std::to_string(src[k]) +
                    " dst=" + std::to_string(dst[k]) + " n_model=" + std::to_string(n_model) +
                    " nlocal=" + std::to_string(nlocal) + " ntotal=" + std::to_string(ntotal))
                       .c_str());
      }
    }
  };
  if (std::getenv("MFF_VALIDATE_GRAPH")) {
    validate_edge_bounds(buf_edge_src_cpu_, buf_edge_dst_cpu_, "main");
    if (use_dispersion_edges) {
      validate_edge_bounds(buf_disp_edge_src_cpu_, buf_disp_edge_dst_cpu_, "dispersion");
    }
  }
  if (std::getenv("MFF_DUMP_GRAPH")) {
    std::ofstream dbg("/tmp/mff_pair_edges.txt");
    dbg << "n_model " << n_model << " nlocal " << nlocal << " ntotal " << ntotal
        << " E " << E << " Edisp " << Edisp << "\n";
    dbg << "atoms\n";
    for (int i = 0; i < n_model; ++i) {
      dbg << i << " " << buf_A_cpu_[static_cast<size_t>(i)] << " "
          << buf_pos_cpu_[static_cast<size_t>(i) * 3 + 0] << " "
          << buf_pos_cpu_[static_cast<size_t>(i) * 3 + 1] << " "
          << buf_pos_cpu_[static_cast<size_t>(i) * 3 + 2] << "\n";
    }
    dbg << "main\n";
    for (size_t k = 0; k < std::min<size_t>(buf_edge_src_cpu_.size(), 64); ++k) {
      dbg << k << " " << buf_edge_src_cpu_[k] << " " << buf_edge_dst_cpu_[k] << " "
          << buf_edge_shifts_cpu_[3 * k + 0] << " " << buf_edge_shifts_cpu_[3 * k + 1] << " "
          << buf_edge_shifts_cpu_[3 * k + 2] << "\n";
    }
    dbg << "dispersion\n";
    for (size_t k = 0; k < std::min<size_t>(buf_disp_edge_src_cpu_.size(), 64); ++k) {
      dbg << k << " " << buf_disp_edge_src_cpu_[k] << " " << buf_disp_edge_dst_cpu_[k] << " "
          << buf_disp_edge_shifts_cpu_[3 * k + 0] << " " << buf_disp_edge_shifts_cpu_[3 * k + 1] << " "
          << buf_disp_edge_shifts_cpu_[3 * k + 2] << "\n";
    }
  }

  // Reuse persistent torch tensors; only reallocate when sizes change.
  if (cached_compute_ntotal_ != static_cast<int64_t>(n_model)) {
    cached_pos_t_ = torch::empty({n_model, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    cached_A_t_ = torch::empty({n_model}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    cached_compute_ntotal_ = static_cast<int64_t>(n_model);
  }
  if (cached_compute_nedges_ != E) {
    cached_edge_src_t_ = torch::empty({E}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    cached_edge_dst_t_ = torch::empty({E}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    cached_edge_shifts_t_ = torch::empty({E, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    cached_compute_nedges_ = E;
  }
  if (cached_compute_disp_nedges_ != Edisp) {
    cached_disp_edge_src_t_ = torch::empty({Edisp}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    cached_disp_edge_dst_t_ = torch::empty({Edisp}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    cached_disp_edge_shifts_t_ = torch::empty({Edisp, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    cached_compute_disp_nedges_ = Edisp;
  }
  std::memcpy(cached_pos_t_.data_ptr<float>(), buf_pos_cpu_.data(),
              static_cast<size_t>(n_model) * 3 * sizeof(float));
  std::memcpy(cached_A_t_.data_ptr<int64_t>(), buf_A_cpu_.data(),
              static_cast<size_t>(n_model) * sizeof(int64_t));
  std::memcpy(cached_edge_src_t_.data_ptr<int64_t>(), buf_edge_src_cpu_.data(),
              static_cast<size_t>(E) * sizeof(int64_t));
  std::memcpy(cached_edge_dst_t_.data_ptr<int64_t>(), buf_edge_dst_cpu_.data(),
              static_cast<size_t>(E) * sizeof(int64_t));
  std::memcpy(cached_edge_shifts_t_.data_ptr<float>(), buf_edge_shifts_cpu_.data(),
              static_cast<size_t>(E) * 3 * sizeof(float));
  if (Edisp > 0) {
    std::memcpy(cached_disp_edge_src_t_.data_ptr<int64_t>(), buf_disp_edge_src_cpu_.data(),
                static_cast<size_t>(Edisp) * sizeof(int64_t));
    std::memcpy(cached_disp_edge_dst_t_.data_ptr<int64_t>(), buf_disp_edge_dst_cpu_.data(),
                static_cast<size_t>(Edisp) * sizeof(int64_t));
    std::memcpy(cached_disp_edge_shifts_t_.data_ptr<float>(), buf_disp_edge_shifts_cpu_.data(),
                static_cast<size_t>(Edisp) * 3 * sizeof(float));
  }
  auto external_tensor_t = current_external_tensor(torch::kCPU);
  auto fidelity_ids_t = current_fidelity_tensor(torch::kCPU);

  const bool want_atom_virial = static_cast<bool>(vflag_atom);
  const bool need_energy = static_cast<bool>(eflag);
  mfftorch::MFFOutputs out;
  const auto profile_pack = profile_pair ? ProfileClock::now() : ProfileClock::time_point{};
  try {
    out = engine_->compute(nlocal, n_model, cached_pos_t_, cached_A_t_,
                           cached_edge_src_t_, cached_edge_dst_t_, cached_edge_shifts_t_,
                           cell_t, cached_disp_edge_src_t_, cached_disp_edge_dst_t_, cached_disp_edge_shifts_t_,
                           external_tensor_t, fidelity_ids_t,
                           need_energy, want_atom_virial);
  } catch (const std::exception &e) {
    if(layerwise_exchange_ && comm->nprocs>1)
      error->one(FLERR, (std::string("mff/torch engine compute failed: ") + e.what()).c_str());
    else error->all(FLERR, (std::string("mff/torch engine compute failed: ") + e.what()).c_str());
  }
  const auto profile_engine = profile_pair ? ProfileClock::now() : ProfileClock::time_point{};
  cache_physical_outputs(out, nlocal);

  mfftorch::ReciprocalOutputs reciprocal_out;
  const bool exports_runtime_source = engine_->exports_reciprocal_source() && engine_->reciprocal_source_channels() > 0;

  const bool has_electrostatics = exports_runtime_source;
  const bool use_reciprocal =
      reciprocal_solver_ && has_electrostatics;
  const bool use_runtime_long_range = use_reciprocal;
  if (use_reciprocal) {
    try {
      const auto reciprocal_device = engine_->device();
      const auto& reciprocal_rows = out.reciprocal_source;
      auto local_source = reciprocal_rows.defined()
                              ? reciprocal_rows.narrow(0, 0, nlocal).to(reciprocal_device, torch::kFloat32).contiguous()
                              : torch::zeros(
                                    {nlocal, engine_->reciprocal_source_channels()},
                                    torch::TensorOptions().dtype(torch::kFloat32).device(reciprocal_device));

      auto reciprocal_inputs = make_reciprocal_inputs(
          world,
          cached_pos_t_.narrow(0, 0, nlocal).to(reciprocal_device, torch::kFloat32).contiguous(),
          local_source,
          cell_t.to(reciprocal_device, torch::kFloat32).contiguous(),
          geom,
          static_cast<bool>(eflag),
          reciprocal_device);
	      if (std::getenv("MFF_RECIPROCAL_MULTIPOLE_AUTOGRAD")) {
	        reciprocal_out = reciprocal_solver_->compute(reciprocal_inputs);
	      } else {
	        c10::InferenceMode reciprocal_inference_guard(true);
	        reciprocal_out = reciprocal_solver_->compute(reciprocal_inputs);
	      }
	      if (std::getenv("MFF_SYNC_AFTER_RECIPROCAL") && reciprocal_device.is_cuda()) {
	        torch::cuda::synchronize();
	      }
	    } catch (const std::exception &e) {
	      error->all(FLERR, (std::string("mff/torch runtime long-range solver failed: ") + e.what()).c_str());
	    }
  }

  if (eflag) eng_vdwl += out.energy;
  if (use_runtime_long_range) eng_vdwl += reciprocal_out.energy;

  // When virial is needed, ghost forces must be in f[] for virial_fdotr_compute()
  // to produce correct results (it sums over nall = nlocal + nghost). In fold mode the model has no
  // ghost nodes (cross-boundary forces already landed on local owners), so we only write nlocal rows.
  const int nwrite = fold_mode_ ? nlocal
      : ((force->newton_pair || vflag_fdotr) ? ntotal : nlocal);
  auto forces_cpu = out.forces.to(torch::kCPU, torch::kFloat64).contiguous();
  const double *fp = forces_cpu.data_ptr<double>();
  for (int i = 0; i < nwrite; i++) {
    f[i][0] += fp[i * 3 + 0];
    f[i][1] += fp[i * 3 + 1];
    f[i][2] += fp[i * 3 + 2];
  }
  if(comm->nprocs>1 && !force->newton_pair)reverse_ghost_contributions(out.forces,ntotal);
  if (use_runtime_long_range && reciprocal_out.forces_local.defined()) {
    auto reciprocal_forces_cpu = reciprocal_out.forces_local.to(torch::kCPU, torch::kFloat64).contiguous();
    const double *rfp = reciprocal_forces_cpu.data_ptr<double>();
    for (int i = 0; i < nlocal; i++) {
      f[i][0] += rfp[i * 3 + 0];
      f[i][1] += rfp[i * 3 + 1];
      f[i][2] += rfp[i * 3 + 2];
    }
  }

  if (eflag_atom && eatom && out.atom_energy.defined()) {
    auto ae_cpu = out.atom_energy.to(torch::kCPU, torch::kFloat64).contiguous().view({n_model});
    const double *ep = ae_cpu.data_ptr<double>();
    for (int i = 0; i < nlocal; i++) eatom[i] += ep[i];
  }
  if (eflag_atom && eatom && use_runtime_long_range && reciprocal_out.atom_energy_local.defined()) {
    auto ae_recip = reciprocal_out.atom_energy_local.to(torch::kCPU, torch::kFloat64).contiguous();
    const double *ep = ae_recip.data_ptr<double>();
    for (int i = 0; i < nlocal; i++) eatom[i] += ep[i];
  }

  if (vflag_atom && vatom && out.atom_virial.defined()) {
    auto vir_cpu = out.atom_virial.to(torch::kCPU, torch::kFloat64).contiguous();
    const double *vp = vir_cpu.data_ptr<double>();
    const int nvir = fold_mode_ ? nlocal : (force->newton_pair ? ntotal : nlocal);
    for (int i = 0; i < nvir; i++) {
      vatom[i][0] += vp[i * 6 + 0];
      vatom[i][1] += vp[i * 6 + 1];
      vatom[i][2] += vp[i * 6 + 2];
      vatom[i][3] += vp[i * 6 + 3];
      vatom[i][4] += vp[i * 6 + 4];
      vatom[i][5] += vp[i * 6 + 5];
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
  if (profile_pair) {
    const auto profile_end = ProfileClock::now();
    const auto elapsed = [](ProfileClock::time_point a, ProfileClock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    // Host wall times, with no additional GPU barrier. The tail includes force
    // readback and any device work still pending when engine_->compute returned.
    std::fprintf(stderr,
        "[MFF_PROFILE_PAIR] step=%lld nodes=%d edges=%lld graph_ms=%.3f sort_ms=%.3f "
        "pack_ms=%.3f engine_host_ms=%.3f tail_ms=%.3f total_ms=%.3f\n",
        static_cast<long long>(update->ntimestep), n_model, static_cast<long long>(E),
        elapsed(profile_start, profile_graph), elapsed(profile_graph, profile_sort),
        elapsed(profile_sort, profile_pack), elapsed(profile_pack, profile_engine),
        elapsed(profile_engine, profile_end), elapsed(profile_start, profile_end));
  }
}
