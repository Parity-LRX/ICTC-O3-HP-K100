#include "mff_cufft_multipole.h"

#include <cufft.h>

#include <cstdio>
#include <cmath>

namespace mfftorch {

namespace {

constexpr float kPi = 3.14159265358979323846f;

__device__ __forceinline__ cufftComplex cadd(cufftComplex a, cufftComplex b) {
  return make_cuFloatComplex(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ cufftComplex csub(cufftComplex a, cufftComplex b) {
  return make_cuFloatComplex(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ cufftComplex cscale(cufftComplex a, float s) {
  return make_cuFloatComplex(a.x * s, a.y * s);
}

__device__ __forceinline__ float sinc_pi(float x) {
  const float ax = fabsf(x);
  if (ax < 1.0e-7f) return 1.0f;
  const float pix = kPi * x;
  return sinf(pix) / pix;
}

__device__ __forceinline__ int wrap_index(int x, int m) {
  x %= m;
  return x < 0 ? x + m : x;
}

__device__ __forceinline__ int freq_index(int i, int m) {
  const int split = (m + 1) / 2;
  return i < split ? i : i - m;
}

// Per-axis mesh assignment: fills the integer base grid index and the stencil weights w[0..S-1]
// for the in-cell scaled coordinate `s` (atom position scaled into [0, mesh)). Mirrors the Python
// reference _assignment_kernel_1d in mace_ictc/models/long_range.py: stencil 2 = CIC (linear,
// base = floor(s)); stencil 4 = PCS (cubic B-spline, base = floor(s) - 1). S <= 4.
__device__ __forceinline__ void assign_weights_1d(float s, int stencil, int& base, float w[4]) {
  if (stencil >= 4) {  // PCS cubic B-spline (C^2 weight, C^1 force, no mesh-cell discontinuity)
    const float fl = floorf(s);
    base = static_cast<int>(fl) - 1;
    const float t = s - fl;  // frac in [0,1)
    const float t2 = t * t, t3 = t2 * t;
    const float omt = 1.0f - t;
    w[0] = (omt * omt * omt) / 6.0f;
    w[1] = (3.0f * t3 - 6.0f * t2 + 4.0f) / 6.0f;
    w[2] = (-3.0f * t3 + 3.0f * t2 + 3.0f * t + 1.0f) / 6.0f;
    w[3] = t3 / 6.0f;
  } else {  // CIC linear 2-point stencil
    base = static_cast<int>(floorf(s));
    const float t = s - static_cast<float>(base);
    w[0] = 1.0f - t;
    w[1] = t;
  }
}

// Stencil size (number of grid points per axis) for a given assignment order. 2=CIC, 4=PCS.
__host__ __device__ __forceinline__ int assignment_stencil(int order) { return order >= 4 ? 4 : 2; }

__global__ void zero_complex_kernel(cufftComplex* data, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  data[i].x = 0.0f;
  data[i].y = 0.0f;
}

__global__ void build_kspec_kernel(
    float* kspec,
    int mesh,
    const float* cell,
    const float* inv_cell,
    float volume,
    float k_norm_floor,
    float ewald_alpha_prefactor,
    int full_ewald,
    int stencil) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int K = mesh * mesh * mesh;
  if (k >= K) return;
  const int z = k % mesh;
  const int y = (k / mesh) % mesh;
  const int x = k / (mesh * mesh);
  const float mx = static_cast<float>(freq_index(x, mesh));
  const float my = static_cast<float>(freq_index(y, mesh));
  const float mz = static_cast<float>(freq_index(z, mesh));

  const float kx = 2.0f * kPi * (mx * inv_cell[0] + my * inv_cell[1] + mz * inv_cell[2]);
  const float ky = 2.0f * kPi * (mx * inv_cell[3] + my * inv_cell[4] + mz * inv_cell[5]);
  const float kz = 2.0f * kPi * (mx * inv_cell[6] + my * inv_cell[7] + mz * inv_cell[8]);
  const float k2 = kx * kx + ky * ky + kz * kz;
  const float kn = sqrtf(k2);

  float spectral = 0.0f;
  if (kn > k_norm_floor) {
    const float sx = sinc_pi(mx / static_cast<float>(mesh));
    const float sy = sinc_pi(my / static_cast<float>(mesh));
    const float sz = sinc_pi(mz / static_cast<float>(mesh));
    // Assignment deconvolution window = (sinc_x sinc_y sinc_z)^stencil (Python _assignment_window_1d
    // with exponent = stencil size), deconvolved by 1/window^2. CIC (stencil 2) -> 1/sinc^4;
    // PCS (stencil 4) -> 1/sinc^8.
    const float s3 = sx * sy * sz;
    float window = s3 * s3;             // stencil 2 (CIC)
    if (stencil >= 4) window *= window; // stencil 4 (PCS): (s3^2)^2 = s3^4
    window = fmaxf(window, 1.0e-6f);
    spectral = (4.0f * kPi) / fmaxf(k2, k_norm_floor * k_norm_floor) / volume / (window * window);
    if (full_ewald) {
      const float r0 = sqrtf(cell[0] * cell[0] + cell[1] * cell[1] + cell[2] * cell[2]);
      const float r1 = sqrtf(cell[3] * cell[3] + cell[4] * cell[4] + cell[5] * cell[5]);
      const float r2 = sqrtf(cell[6] * cell[6] + cell[7] * cell[7] + cell[8] * cell[8]);
      const float lmin = fminf(r0, fminf(r1, r2));
      const float real_cutoff = fmaxf(0.5f * lmin, k_norm_floor);
      const float alpha = ewald_alpha_prefactor / real_cutoff;
      spectral *= expf(-k2 / (4.0f * alpha * alpha));
    }
  }
  kspec[4 * k + 0] = kx;
  kspec[4 * k + 1] = ky;
  kspec[4 * k + 2] = kz;
  kspec[4 * k + 3] = spectral;
}

__global__ void spread_source_kernel(
    const float* pos,
    const float* source,
    const float* inv_cell,
    cufftComplex* mesh_data,
    int n_atoms,
    int mesh,
    int packed_width,
    int stencil) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_atoms * packed_width;
  if (idx >= total) return;
  const int atom = idx / packed_width;
  const int ch = idx - atom * packed_width;
  const float q = source[idx];
  if (q == 0.0f) return;

  float fx = pos[3 * atom + 0] * inv_cell[0] + pos[3 * atom + 1] * inv_cell[3] + pos[3 * atom + 2] * inv_cell[6];
  float fy = pos[3 * atom + 0] * inv_cell[1] + pos[3 * atom + 1] * inv_cell[4] + pos[3 * atom + 2] * inv_cell[7];
  float fz = pos[3 * atom + 0] * inv_cell[2] + pos[3 * atom + 1] * inv_cell[5] + pos[3 * atom + 2] * inv_cell[8];
  fx -= floorf(fx);
  fy -= floorf(fy);
  fz -= floorf(fz);

  const float sx = fx * mesh;
  const float sy = fy * mesh;
  const float sz = fz * mesh;
  int bx, by, bz;
  float wx[4], wy[4], wz[4];
  assign_weights_1d(sx, stencil, bx, wx);
  assign_weights_1d(sy, stencil, by, wy);
  assign_weights_1d(sz, stencil, bz, wz);
  const int K = mesh * mesh * mesh;
  for (int ix = 0; ix < stencil; ++ix) {
    const int x = wrap_index(bx + ix, mesh);
    for (int iy = 0; iy < stencil; ++iy) {
      const int y = wrap_index(by + iy, mesh);
      for (int iz = 0; iz < stencil; ++iz) {
        const int z = wrap_index(bz + iz, mesh);
        const float w = wx[ix] * wy[iy] * wz[iz];
        const int flat = (x * mesh + y) * mesh + z;
        atomicAdd(&mesh_data[ch * K + flat].x, q * w);
      }
    }
  }
}

__global__ void build_grad_and_energy_kernel(
    const cufftComplex* packed_fft,
    const float* kspec,
    cufftComplex* grad_fft,
    double* energy,
    int mesh,
    int source_channels,
    int max_l,
    float energy_scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int K = mesh * mesh * mesh;
  const int total = K * source_channels;
  if (idx >= total) return;
  const int c = idx / K;
  const int k = idx - c * K;
  const int C = source_channels;
  const int P = C * (1 + (max_l >= 1 ? 3 : 0) + (max_l >= 2 ? 9 : 0));
  const float kx = kspec[4 * k + 0];
  const float ky = kspec[4 * k + 1];
  const float kz = kspec[4 * k + 2];
  const float spectral = kspec[4 * k + 3];

  cufftComplex S = packed_fft[c * K + k];
  if (max_l >= 1) {
    const cufftComplex mux = packed_fft[(C + 3 * c + 0) * K + k];
    const cufftComplex muy = packed_fft[(C + 3 * c + 1) * K + k];
    const cufftComplex muz = packed_fft[(C + 3 * c + 2) * K + k];
    cufftComplex dot = cadd(cadd(cscale(mux, kx), cscale(muy, ky)), cscale(muz, kz));
    S = cadd(S, make_cuFloatComplex(-dot.y, dot.x));
  }
  if (max_l >= 2) {
    cufftComplex qsum = make_cuFloatComplex(0.0f, 0.0f);
    const float kv[3] = {kx, ky, kz};
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        qsum = cadd(qsum, cscale(packed_fft[(4 * C + 9 * c + 3 * a + b) * K + k], kv[a] * kv[b]));
      }
    }
    S = csub(S, cscale(qsum, 0.5f));
  }
  if (spectral != 0.0f) {
    atomicAdd(energy, static_cast<double>(0.5f * energy_scale * spectral * (S.x * S.x + S.y * S.y)));
  }

  const cufftComplex base = cscale(S, energy_scale * spectral);
  cufftComplex field[13];
  int nfield = 0;
  field[nfield++] = base;
  if (max_l >= 1) {
    field[nfield++] = make_cuFloatComplex(base.y * kx, -base.x * kx);  // base * (-i kx)
    field[nfield++] = make_cuFloatComplex(base.y * ky, -base.x * ky);
    field[nfield++] = make_cuFloatComplex(base.y * kz, -base.x * kz);
  }
  if (max_l >= 2) {
    const float kv[3] = {kx, ky, kz};
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        field[nfield++] = cscale(base, -0.5f * kv[a] * kv[b]);
      }
    }
  }
  for (int p_local = 0; p_local < nfield; ++p_local) {
    const int p = c * nfield + p_local;
    const cufftComplex f = field[p_local];
    const cufftComplex gx = make_cuFloatComplex(-f.y * kx, f.x * kx);  // i kx f
    const cufftComplex gy = make_cuFloatComplex(-f.y * ky, f.x * ky);
    const cufftComplex gz = make_cuFloatComplex(-f.y * kz, f.x * kz);
    grad_fft[(0 * P + p) * K + k] = gx;
    grad_fft[(1 * P + p) * K + k] = gy;
    grad_fft[(2 * P + p) * K + k] = gz;
  }
}

__global__ void gather_force_kernel(
    const float* pos,
    const float* source,
    const float* inv_cell,
    const cufftComplex* grad_mesh,
    float* forces,
    int n_atoms,
    int mesh,
    int packed_width,
    int stencil) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= n_atoms) return;
  float fx = pos[3 * atom + 0] * inv_cell[0] + pos[3 * atom + 1] * inv_cell[3] + pos[3 * atom + 2] * inv_cell[6];
  float fy = pos[3 * atom + 0] * inv_cell[1] + pos[3 * atom + 1] * inv_cell[4] + pos[3 * atom + 2] * inv_cell[7];
  float fz = pos[3 * atom + 0] * inv_cell[2] + pos[3 * atom + 1] * inv_cell[5] + pos[3 * atom + 2] * inv_cell[8];
  fx -= floorf(fx);
  fy -= floorf(fy);
  fz -= floorf(fz);
  const float sx = fx * mesh;
  const float sy = fy * mesh;
  const float sz = fz * mesh;
  int bx, by, bz;
  float wx[4], wy[4], wz[4];
  assign_weights_1d(sx, stencil, bx, wx);
  assign_weights_1d(sy, stencil, by, wy);
  assign_weights_1d(sz, stencil, bz, wz);
  const int K = mesh * mesh * mesh;
  float out[3] = {0.0f, 0.0f, 0.0f};
  for (int ix = 0; ix < stencil; ++ix) {
    const int x = wrap_index(bx + ix, mesh);
    for (int iy = 0; iy < stencil; ++iy) {
      const int y = wrap_index(by + iy, mesh);
      for (int iz = 0; iz < stencil; ++iz) {
        const int z = wrap_index(bz + iz, mesh);
        const float w = wx[ix] * wy[iy] * wz[iz];
        const int flat = (x * mesh + y) * mesh + z;
        for (int p = 0; p < packed_width; ++p) {
          const float q = source[atom * packed_width + p];
          out[0] -= q * w * grad_mesh[(0 * packed_width + p) * K + flat].x;
          out[1] -= q * w * grad_mesh[(1 * packed_width + p) * K + flat].x;
          out[2] -= q * w * grad_mesh[(2 * packed_width + p) * K + flat].x;
        }
      }
    }
  }
  forces[3 * atom + 0] = out[0];
  forces[3 * atom + 1] = out[1];
  forces[3 * atom + 2] = out[2];
}

bool fail(char* error_msg, int error_msg_len, const char* msg) {
  if (error_msg && error_msg_len > 0) {
    std::snprintf(error_msg, static_cast<size_t>(error_msg_len), "%s", msg);
  }
  return false;
}

bool check_cuda(cudaError_t err, char* error_msg, int error_msg_len, const char* what) {
  if (err == cudaSuccess) return true;
  if (error_msg && error_msg_len > 0) {
    std::snprintf(error_msg, static_cast<size_t>(error_msg_len), "%s: %s", what, cudaGetErrorString(err));
  }
  return false;
}

bool check_cufft(cufftResult err, char* error_msg, int error_msg_len, const char* what) {
  if (err == CUFFT_SUCCESS) return true;
  if (error_msg && error_msg_len > 0) {
    std::snprintf(error_msg, static_cast<size_t>(error_msg_len), "%s: cufft error %d", what, static_cast<int>(err));
  }
  return false;
}

}  // namespace

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
    int error_msg_len) {
  if (params.n_atoms < 0 || params.mesh <= 0 || params.source_channels <= 0) {
    return fail(error_msg, error_msg_len, "invalid cuFFT multipole dimensions");
  }
  if (params.pbc[0] != 1 || params.pbc[1] != 1 || params.pbc[2] != 1) {
    return fail(error_msg, error_msg_len, "cuFFT multipole path currently requires 3D periodic PBC");
  }
  if (params.max_multipole_l < 0 || params.max_multipole_l > 2) {
    return fail(error_msg, error_msg_len, "cuFFT multipole path supports max_multipole_l in [0,2]");
  }
  const int n_atoms = params.n_atoms;
  const int mesh = params.mesh;
  const int C = params.source_channels;
  const int packed_width = C * (1 + (params.max_multipole_l >= 1 ? 3 : 0) + (params.max_multipole_l >= 2 ? 9 : 0));
  const int K = mesh * mesh * mesh;
  // Mesh assignment stencil (2=CIC, 4=PCS), matching the in-model long_range_assignment so the
  // deployed spreading/deconvolution reproduces the training-time PME exactly.
  const int stencil = assignment_stencil(params.assignment_order);
  cufftComplex* mesh_complex = static_cast<cufftComplex*>(workspace.mesh_complex);
  cufftComplex* grad_complex = static_cast<cufftComplex*>(workspace.grad_complex);
  if (!mesh_complex || !grad_complex || !workspace.kspec || !workspace.energy) {
    return fail(error_msg, error_msg_len, "cuFFT multipole workspace is incomplete");
  }

  constexpr int block = 256;
  zero_complex_kernel<<<(packed_width * K + block - 1) / block, block, 0, stream>>>(mesh_complex, packed_width * K);
  zero_complex_kernel<<<(3 * packed_width * K + block - 1) / block, block, 0, stream>>>(
      grad_complex, 3 * packed_width * K);
  if (!check_cuda(cudaMemsetAsync(workspace.energy, 0, sizeof(double), stream), error_msg, error_msg_len, "cudaMemset energy")) {
    return false;
  }
  build_kspec_kernel<<<(K + block - 1) / block, block, 0, stream>>>(
      workspace.kspec,
      mesh,
      cell,
      inv_cell,
      params.volume,
      params.k_norm_floor,
      params.ewald_alpha_prefactor,
      params.full_ewald ? 1 : 0,
      stencil);
  spread_source_kernel<<<(n_atoms * packed_width + block - 1) / block, block, 0, stream>>>(
      pos, packed_source, inv_cell, mesh_complex, n_atoms, mesh, packed_width, stencil);
  if (!check_cuda(cudaGetLastError(), error_msg, error_msg_len, "cuFFT multipole spread launch")) return false;

  cufftHandle plan_forward = 0;
  int n[3] = {mesh, mesh, mesh};
  if (!check_cufft(
          cufftPlanMany(&plan_forward, 3, n, nullptr, 1, K, nullptr, 1, K, CUFFT_C2C, packed_width),
          error_msg,
          error_msg_len,
          "cufftPlanMany forward")) {
    return false;
  }
  cufftSetStream(plan_forward, stream);
  if (!check_cufft(cufftExecC2C(plan_forward, mesh_complex, mesh_complex, CUFFT_FORWARD), error_msg, error_msg_len, "cufftExec forward")) {
    cufftDestroy(plan_forward);
    return false;
  }
  cufftDestroy(plan_forward);

  build_grad_and_energy_kernel<<<(K * C + block - 1) / block, block, 0, stream>>>(
      mesh_complex,
      workspace.kspec,
      grad_complex,
      workspace.energy,
      mesh,
      C,
      params.max_multipole_l,
      params.energy_scale);
  if (!check_cuda(cudaGetLastError(), error_msg, error_msg_len, "cuFFT multipole grad launch")) return false;

  cufftHandle plan_inverse = 0;
  if (!check_cufft(
          cufftPlanMany(&plan_inverse, 3, n, nullptr, 1, K, nullptr, 1, K, CUFFT_C2C, 3 * packed_width),
          error_msg,
          error_msg_len,
          "cufftPlanMany inverse")) {
    return false;
  }
  cufftSetStream(plan_inverse, stream);
  if (!check_cufft(cufftExecC2C(plan_inverse, grad_complex, grad_complex, CUFFT_INVERSE), error_msg, error_msg_len, "cufftExec inverse")) {
    cufftDestroy(plan_inverse);
    return false;
  }
  cufftDestroy(plan_inverse);

  gather_force_kernel<<<(n_atoms + block - 1) / block, block, 0, stream>>>(
      pos, packed_source, inv_cell, grad_complex, forces, n_atoms, mesh, packed_width, stencil);
  return check_cuda(cudaGetLastError(), error_msg, error_msg_len, "cuFFT multipole gather launch");
}

}  // namespace mfftorch
