# USER-MFFTORCH package: LibTorch-powered pair style(s)
#
# This file must live under LAMMPS/cmake/Packages/ so it can be included via:
#   include(Packages/USER-MFFTORCH)
#
# It wires LibTorch include dirs + link libs into the `lammps` target so that
# style headers included from generated style_pair.h can compile.

find_package(Torch REQUIRED)
option(MFF_USE_HIP "Build mff/torch against HIP-enabled LibTorch" OFF)
if(MFF_USE_HIP)
  find_package(hip REQUIRED CONFIG)
  target_compile_definitions(lammps PRIVATE MFF_USE_HIP=1 USE_ROCM=1)
  target_link_libraries(lammps PRIVATE hip::host)
else()
  find_package(CUDAToolkit QUIET)
endif()

option(MFF_ENABLE_VIRIAL "Enable virial/stress calculation in mff/torch Kokkos pair style" OFF)
if(MFF_ENABLE_VIRIAL)
  target_compile_definitions(lammps PRIVATE MFF_ENABLE_VIRIAL)
endif()

target_sources(lammps PRIVATE
  ${LAMMPS_SOURCE_DIR}/USER-MFFTORCH/mff_torch_engine.cpp
  ${LAMMPS_SOURCE_DIR}/USER-MFFTORCH/mff_reciprocal_solver.cpp
  ${LAMMPS_SOURCE_DIR}/USER-MFFTORCH/mff_mbd_solver.cpp
  ${LAMMPS_SOURCE_DIR}/USER-MFFTORCH/mff_tree_fmm_solver.cpp
  ${LAMMPS_SOURCE_DIR}/USER-MFFTORCH/compute_mff_torch_phys.cpp
  ${LAMMPS_SOURCE_DIR}/USER-MFFTORCH/pair_mff_torch.cpp
)

if(CUDAToolkit_FOUND AND NOT MFF_USE_HIP)
  enable_language(CUDA)
  target_sources(lammps PRIVATE
    ${LAMMPS_SOURCE_DIR}/USER-MFFTORCH/mff_cufft_multipole.cu
  )
  target_compile_definitions(lammps PRIVATE MFF_HAS_CUFFT=1)
  target_link_libraries(lammps PRIVATE CUDA::cufft CUDA::cudart)
endif()

if(PKG_KOKKOS)
  target_sources(lammps PRIVATE
    ${LAMMPS_SOURCE_DIR}/USER-MFFTORCH/pair_mff_torch_kokkos.cpp
  )
endif()

if(TARGET Torch::Torch)
  target_link_libraries(lammps PRIVATE Torch::Torch)
else()
  # Fallback for older TorchConfig.cmake
  target_include_directories(lammps PRIVATE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(lammps PRIVATE "${TORCH_LIBRARIES}")
  if(TORCH_CXX_FLAGS)
    separate_arguments(_TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(lammps PRIVATE ${_TORCH_CXX_FLAGS_LIST})
  endif()
endif()
