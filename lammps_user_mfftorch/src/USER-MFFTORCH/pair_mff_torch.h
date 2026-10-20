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
class MFFMBDSolver;
class MFFTreeFmmSolver;
struct MFFOutputs;
}

namespace LAMMPS_NS {

class PairMFFTorch : public Pair {
 public:
  PairMFFTorch(class LAMMPS *lmp);
  ~PairMFFTorch() override;

  void compute(int eflag, int vflag) override;
  // rRESPA multiple-time-stepping: the cheap short-range (SR) core runs on the INNER level (every
  // step) while the expensive many-body-dispersion (MBD) solve runs on the OUTER level (every K
  // steps). The dispersion force is smooth/slow so K=2-4 conserves energy well. See compute_inner/
  // compute_outer below; the non-respa compute() above is unchanged (runs both every step).
  void compute_inner() override;
  void compute_middle() override;
  void compute_outer(int eflag, int vflag) override;
  void settings(int narg, char **arg) override;
  void coeff(int narg, char **arg) override;
  double init_one(int i, int j) override;
  void init_style() override;
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
  std::unique_ptr<mfftorch::MFFMBDSolver> mbd_solver_;
  std::unique_ptr<mfftorch::MFFTreeFmmSolver> tree_fmm_solver_;
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
  int64_t cached_compute_nedges_ = 0;
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

  // --- rRESPA support -----------------------------------------------------------------------------
  // compute() is parameterized by respa_phase_ so the inner/outer levels reuse its (large) graph
  // builder + force/energy bookkeeping instead of duplicating it:
  //   PHASE_FULL  (non-respa / run_style verlet): build graph, SR forward, reciprocal, MBD, tally all,
  //               virial_fdotr_compute. Unchanged from before respa.
  //   PHASE_INNER (compute_inner, every step): build graph, SR forward, add SR forces, run reciprocal,
  //               STAGE the MBD inputs (reciprocal_source + disp edges + cell + pos). NO MBD solve, NO
  //               energy/virial tally (deferred to the outer level).
  //   PHASE_OUTER (compute_outer, every K steps): SKIP graph rebuild + SR forward; run the MBD solve
  //               from the staged inputs, add MBD forces, and tally the staged SR energy + MBD energy.
  enum RespaPhase { PHASE_FULL = 0, PHASE_INNER = 1, PHASE_OUTER = 2 };
  int respa_phase_ = PHASE_FULL;

  // Staged-for-outer state (filled in PHASE_INNER, consumed in PHASE_OUTER).
  bool respa_staged_ = false;            // PHASE_INNER produced usable staged state this step
  double respa_sr_energy_ = 0.0;         // staged SR scalar energy (out.energy)
  int respa_staged_nlocal_ = 0;          // nlocal at stage time
  bool respa_have_mbd_ = false;          // an MBD source/edges were staged this inner step
  torch::Tensor respa_mbd_source_;       // staged MBD source [nlocal,chan] on engine device
  torch::Tensor respa_mbd_pos_;          // staged positions [nlocal,3] on engine device
  torch::Tensor respa_mbd_cell_;         // staged cell [3,3] on engine device
  torch::Tensor respa_mbd_full_src_;     // staged full (both-dir) dispersion edge src (engine device)
  torch::Tensor respa_mbd_full_dst_;     // staged full dispersion edge dst
  torch::Tensor respa_mbd_full_sh_;      // staged full dispersion edge shifts
  bool respa_mbd_have_edges_ = false;    // staged a LAMMPS dispersion edge list (vs solver fallback)
};

}  // namespace LAMMPS_NS

#endif  // LMP_PAIR_MFF_TORCH_H
#endif  // PAIR_CLASS
