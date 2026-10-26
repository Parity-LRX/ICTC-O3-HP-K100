#ifdef PAIR_CLASS
// clang-format off
PairStyle(mff/torch,PairMFFTorch);
// clang-format on
#else

#ifndef LMP_PAIR_MFF_TORCH_H
#define LMP_PAIR_MFF_TORCH_H

#include "pair.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace mfftorch {
class MFFTorchEngine;
class MFFReciprocalSolver;
struct MFFOutputs;
}

namespace LAMMPS_NS {

class PairMFFTorch : public Pair {
 public:
  PairMFFTorch(class LAMMPS *lmp);
  ~PairMFFTorch() override;

  void compute(int eflag, int vflag) override;
  void settings(int narg, char **arg) override;
  void coeff(int narg, char **arg) override;
  double init_one(int i, int j) override;
  void init_style() override;
  int pack_forward_comm(int n,int *list,double *buf,int pbc_flag,int *pbc) override;
  void unpack_forward_comm(int n,int first,double *buf) override;
  int pack_reverse_comm(int n,int first,double *buf) override;
  void unpack_reverse_comm(int n,int *list,double *buf) override;
  void set_physical_cache_requested(bool requested) { physical_cache_requested_ = requested; }
  const torch::Tensor& global_phys() const { return global_phys_cpu_; }
  const torch::Tensor& atom_phys() const { return atom_phys_cpu_; }
  const torch::Tensor& global_phys_mask() const { return global_phys_mask_cpu_; }
  const torch::Tensor& atom_phys_mask() const { return atom_phys_mask_cpu_; }
  int64_t cached_phys_timestep() const { return cached_phys_timestep_; }

 protected:
  void allocate();
  torch::Tensor current_external_tensor(const torch::Device& device);
  void validate_external_field_configuration();
  void cache_physical_outputs(const mfftorch::MFFOutputs& out, int nlocal);
  void reset_physical_outputs();
  torch::Tensor current_fidelity_tensor(const torch::Device& device);
  void reverse_ghost_contributions(const torch::Tensor& forces,int ntotal,bool add_to_host=true);
  std::vector<double> reverse_force_buffer_;
  std::vector<double> owner_map_buffer_;
  bool layerwise_exchange_ = false;
  void refresh_domain_owners();

  double cut_global_ = 0.0;
  double cutsq_global_ = 0.0;
  double dispersion_cut_global_ = 0.0;
  double dispersion_cutsq_global_ = 0.0;
  double request_cut_global_ = 0.0;
  double request_cutsq_global_ = 0.0;
  // Message-passing depth (num_interaction). The ghost halo is extended to mp_depth_*cutoff so each
  // local atom's full K-hop environment is present -> correct under MPI domain decomposition.
  int mp_depth_ = 2;
  bool mp_depth_explicit_ = false;
  void configure_message_depth();
  // Single-rank (nprocs==1) fast path: fold periodic-ghost neighbours back to their local owner
  // (atom->map) + an integer cell shift, so the model runs on ONLY the nlocal local atoms (the exact
  // training graph: local nodes + PBC shifts), instead of the mp_depth_*cutoff ghost halo. This cuts
  // the model's node count from ntotal (~12x the locals on a small periodic cell) to nlocal -> ~12x
  // less memory + compute, and is what makes larger systems fit on one GPU. Folding only works on a
  // single subdomain (a boundary ghost's owner is local), so np>1 keeps the refined-A 2x-halo path.
  bool fold_mode_ = false;

  std::string device_str_ = "cuda";
  std::string core_pt_path_;

  std::vector<int64_t> type2Z_;
  bool use_external_field_ = false;
  bool use_electric_field_ = false;
  bool use_magnetic_field_ = false;
  bool use_rank2_external_field_ = false;
  bool external_field_symmetric_rank2_ = false;
  bool use_fidelity_input_ = false;
  bool fidelity_is_variable_ = false;
  std::string fidelity_var_name_;
  int64_t fidelity_constant_ = 0;
  std::vector<std::string> electric_field_var_names_;
  std::vector<std::string> magnetic_field_var_names_;
  std::vector<std::string> rank2_external_field_var_names_;
  std::vector<float> cached_external_field_values_;
  torch::Tensor external_tensor_cache_;
  torch::Tensor global_phys_cpu_;
  torch::Tensor atom_phys_cpu_;
  torch::Tensor global_phys_mask_cpu_;
  torch::Tensor atom_phys_mask_cpu_;
  int64_t cached_phys_timestep_ = -1;
  bool physical_cache_requested_ = false;

  std::unique_ptr<mfftorch::MFFTorchEngine> engine_;
  std::unique_ptr<mfftorch::MFFReciprocalSolver> reciprocal_solver_;
  bool engine_loaded_ = false;

  // Persistent per-step CPU buffers (avoid repeated heap allocation).
  std::vector<int64_t> buf_A_cpu_;
  std::vector<float> buf_pos_cpu_;
  std::vector<int64_t> buf_edge_src_cpu_;
  std::vector<int64_t> buf_edge_dst_cpu_;
  std::vector<float> buf_edge_shifts_cpu_;
  std::vector<int64_t> buf_disp_edge_src_cpu_;
  std::vector<int64_t> buf_disp_edge_dst_cpu_;
  std::vector<float> buf_disp_edge_shifts_cpu_;

  // Persistent torch tensors (avoid from_blob().clone() every step).
  int64_t cached_compute_ntotal_ = 0;
  int64_t cached_compute_nedges_ = -1;
  int64_t cached_compute_disp_nedges_ = 0;
  torch::Tensor cached_pos_t_;
  torch::Tensor cached_A_t_;
  torch::Tensor cached_edge_src_t_;
  torch::Tensor cached_edge_dst_t_;
  torch::Tensor cached_edge_shifts_t_;
  torch::Tensor cached_disp_edge_src_t_;
  torch::Tensor cached_disp_edge_dst_t_;
  torch::Tensor cached_disp_edge_shifts_t_;
  torch::Tensor cached_cell_t_;
};

}  // namespace LAMMPS_NS

#endif  // LMP_PAIR_MFF_TORCH_H
#endif  // PAIR_CLASS
