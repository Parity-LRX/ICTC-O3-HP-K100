#pragma once

#include <cuda_runtime_api.h>

namespace mfftorch {

struct CufftMultipoleParams {
  int n_atoms = 0;
  int mesh = 0;
  int source_channels = 1;
  int max_multipole_l = 2;
  int pbc[3] = {1, 1, 1};
  float volume = 1.0f;
  float k_norm_floor = 1.0e-6f;
  float ewald_alpha_prefactor = 5.0f;
  float energy_scale = 1.0f;
  bool full_ewald = false;
  int assignment_order = 2;  // mesh assignment stencil: 2=CIC, 4=PCS (cubic B-spline)
};

struct CufftMultipoleWorkspace {
  void* mesh_complex = nullptr;      // cufftComplex[P*K]
  void* grad_complex = nullptr;      // cufftComplex[3*P*K]
  float* kspec = nullptr;            // float[K*4] = kx,ky,kz,spectral
  double* energy = nullptr;          // double[1]
};

bool cufft_multipole_compute(
    const CufftMultipoleParams& params,
    const float* pos,
    const float* packed_source,
    const float* cell,
    const float* inv_cell,
    float* forces,
    CufftMultipoleWorkspace workspace,
    cudaStream_t stream,
    char* error_msg,
    int error_msg_len);

}  // namespace mfftorch
