#pragma once

#include <c10/core/DeviceGuard.h>
#include <c10/core/StreamGuard.h>
#include <stdexcept>
#include <string>

#ifdef MFF_USE_HIP
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <hip/hip_runtime_api.h>
#else
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#endif

namespace mfftorch::gpu {
inline int device_count() {
  int count = 0;
#ifdef MFF_USE_HIP
  if (hipGetDeviceCount(&count) != hipSuccess) { hipGetLastError(); return 0; }
#else
  if (cudaGetDeviceCount(&count) != cudaSuccess) { cudaGetLastError(); return 0; }
#endif
  return count;
}

inline void synchronize() {
#ifdef MFF_USE_HIP
  auto error = hipDeviceSynchronize();
  if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
#else
  auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
#endif
}

inline void set_device(int index) {
#ifdef MFF_USE_HIP
  auto error = hipSetDevice(index);
  if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
#else
  c10::cuda::set_device(index);
#endif
}

inline c10::Stream default_stream(int index) {
#ifdef MFF_USE_HIP
  return c10::hip::getDefaultHIPStreamMasqueradingAsCUDA(index);
#else
  return c10::cuda::getDefaultCUDAStream(index);
#endif
}
}  // namespace mfftorch::gpu
