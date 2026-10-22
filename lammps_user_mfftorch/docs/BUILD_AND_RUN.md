# Build and run USER-MFFTORCH on K100

This guide is for Hygon K100 (`gfx926`) with HIP/DTK. The C++ deployment format
is a TorchScript archive. AOTInductor `.pt2`, whole-model `torch.compile` and
graph replay are outside the supported K100 workflow.

The numerical checks use the vendor DTK 26.04 stack, PyTorch 2.7.1 with
HIP 6.3.26093, FP32 arithmetic and TF32 disabled. The default TorchScript archive dispatches to native eager HP kernels; the
portable reference route is a separate export option.

## 1. Environment and execution

Use a Linux environment with:

- The matching HIP-enabled PyTorch/LibTorch, C++ ABI and shared libraries.
- DTK and a C++17 compiler supporting the target `gfx926` architecture.
- CMake and a LAMMPS/Kokkos tree with working K100 support.
- This package and its conversion dependencies for Python conversion/export.

The build used for this guide starts from LAMMPS `22Jul2025_update6` with
vendor-specific `gfx926` Kokkos support. An upstream checkout without that
support is insufficient: setting `MFF_USE_HIP=ON` does not add a missing
Kokkos architecture or compiler target.

Activate the vendor environment, then install this repository:

```bash
python -m pip install -e '.[hp,conversion,lammps]'
```

PyTorch uses `device="cuda"` and `torch.cuda` for K100 HIP devices. These are
API names. Preserve the vendor PyTorch and Triton installations.

On Slurm clusters, compilation, conversion/export, numerical checks and MD
belong in jobs submitted through `sbatch`. Choose the site's K100 partition,
account, GPU request and CPU allocation. Use `srun` only inside that job.
Source checkout, file copying and documentation edits can be prepared first.

## 2. Install the C++ package

From the ICTC-O3-HP-K100 repository root:

```bash
lammps_source=/path/to/lammps
mkdir -p "$lammps_source/src/USER-MFFTORCH"
cp -R lammps_user_mfftorch/src/USER-MFFTORCH/. \
  "$lammps_source/src/USER-MFFTORCH/"
```

Copy the package definition into the layout used by the LAMMPS release.
For `cmake/Modules/Packages`:

```bash
cp lammps_user_mfftorch/cmake/Modules/Packages/USER-MFFTORCH.cmake \
  "$lammps_source/cmake/Modules/Packages/"
```

For a release using `cmake/Packages`, copy the corresponding file from
`lammps_user_mfftorch/cmake/Packages/` instead.

In `LAMMPS/cmake/CMakeLists.txt`, register `USER-MFFTORCH` in
`STANDARD_PACKAGES` and the existing `PKG_WITH_INCL` package loop. If that
loop is absent, include the package definition conditionally:

```cmake
if(PKG_USER-MFFTORCH)
  include(Modules/Packages/USER-MFFTORCH)
endif()
```

Use `include(Packages/USER-MFFTORCH)` for the other layout. Include the
package definition once.

## 3. Build shared MPI/HIP LAMMPS

The default build supports one MPI rank per K100 GPU. Load a consistent MPI
compiler/runtime before configuring LAMMPS. The plain and Kokkos pair styles
are both included. For a serial-only build, use a separate directory and
`BUILD_MPI=OFF`.

Inside the batch job, activate the matching compiler/DTK environment and run:

```bash
cd /path/to/lammps
torch_root="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')"

cmake -S cmake -B build-mfftorch-hip \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=17 \
  -D CMAKE_CXX_COMPILER="$(command -v hipcc)" \
  -D CMAKE_PREFIX_PATH="$torch_root/share/cmake;$ROCM_PATH" \
  -D CMAKE_INSTALL_PREFIX=/path/to/lammps-install \
  -D BUILD_SHARED_LIBS=ON \
  -D BUILD_MPI=ON \
  -D BUILD_OMP=OFF \
  -D PKG_USER-MFFTORCH=ON \
  -D MFF_USE_HIP=ON \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_HIP=ON \
  -D Kokkos_ENABLE_SERIAL=ON \
  -D Kokkos_ENABLE_OPENMP=OFF \
  -D Kokkos_ARCH_AMD_GFX926=ON \
  -D AMDGPU_TARGETS=gfx926 \
  -D GPU_TARGETS=gfx926 \
  -D CMAKE_HIP_ARCHITECTURES=gfx926
cmake --build build-mfftorch-hip --parallel 8
cmake --install build-mfftorch-hip
```

`Kokkos_ARCH_AMD_GFX926` requires the corresponding vendor Kokkos support.
Use the compiler, host-GCC selection and library search options required by
that installation. For example, the tested DTK compiler uses GCC 11 host
libraries; its task configuration supplies
`--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11`.

Some vendor Torch CMake files contain absolute paths from their build machine.
Resolve missing imported libraries in a task-local configuration using the
installed HIP runtime and system libraries. This is an environment
compatibility step, separate from enabling `USER-MFFTORCH`.

The tested wheel records `/opt/dtk/lib/libamdhip64.so` and
`/usr/lib/x86_64-linux-gnu/librt.so`. When those exact paths are absent, the
task-local package definition uses the following adjustment after finding
Torch and HIP:

```cmake
foreach(_target c10_hip torch_hip MIOpen)
  if(TARGET ${_target})
    get_target_property(_links ${_target} INTERFACE_LINK_LIBRARIES)
    if(_links)
      string(REPLACE "/opt/dtk/lib/libamdhip64.so" "hip::host" _links "${_links}")
      string(REPLACE "/usr/lib/x86_64-linux-gnu/librt.so" "rt" _links "${_links}")
      set_property(TARGET ${_target} PROPERTY INTERFACE_LINK_LIBRARIES "${_links}")
    endif()
  endif()
endforeach()
```

Supply the vendor environment's MKL include directory and wheel-library
linker search path as needed. These paths depend on the installation.

For a plain `mff/torch` build, omit the Kokkos options and use a separate build
directory. Match `--parallel` to the allocated CPU cores. HIP builds do not
enable the engine's CUDA Graph replay implementation.

## 4. Convert and export the model

### Standard MACE conversion

For a compatible `ScaleShiftMACE` model:

```bash
mff-convert-mace \
  --mace-model /path/to/original.model \
  --out /path/to/converted.pth \
  --product-backend ictd-bridge-u \
  --dtype float32 --device cpu
```

The converter preserves radial MLP hidden widths and depth, normalized
activations, learned matrices, TP path normalization, energy scale/shift and
E0. Positive integer hidden widths and an empty hidden list are supported.
`radial_mlp_hidden` is recorded in architecture metadata for reconstruction.
Legacy checkpoints without that field use `[64,64,64]`.

Other structural requirements remain: the supported ordinary/residual
interaction classes, at least two interactions, natural-parity contiguous
hidden degrees with equal multiplicity, Bessel radial features, uniform
correlation and SiLU readout. Extra distance transforms, pair repulsion and
nonlinear foundation architectures need their corresponding implementation.
See the [conversion API](../../docs/HP_API.md#6-checkpoint-conversion-and-lammps-on-k100)
and the separate [FoundationHP interface](../../docs/HP_API.md#nonlinear-mace-foundation-checkpoints).

### Native HP export (default)

```bash
mff-export-core --checkpoint /path/to/converted.pth \
  --elements H C O --out /path/to/core.pt
```

This selects `--backend native --inference-mode performance`, FP32 and E0
included. The exporter validates model reconstruction and generated-operator
coverage on K100. It preserves the original checkpoint and records its
architecture, trained cutoff, species, head and inference mode.

For a converted nonlinear foundation model:

```bash
mff-export-core --checkpoint /path/to/mace-mh-1-ictc-hp.pt \
  --elements Cu --head omat_pbe --out /path/to/mh1-core.pt
```

All original heads remain in the checkpoint; one head is evaluated. Use
`--inference-mode capacity` for smaller forward message chunks. Performance
chunks use the model's actual TP widths and GPU memory to bound the largest
message temporary. Backward and radial chunks are bounded separately.

If neighbor normalization is absent from an older checkpoint, supply its
recorded training value with `--avg-num-neighbors`. Keep `core.pt`,
`core.pt.checkpoint.pth`, `core.pt.json` and `core.pt.meta` together. The
checkpoint path in the metadata is relative, so the set can be relocated.

The archive is a scripted dispatch envelope for the installed HP runtime.
It does not trace or compile the whole model. Import and reconstruction are
automatic at C++ load time. A raw checkpoint renamed to `.pt` is insufficient.
`--backend reference` explicitly selects the original portable TorchScript
reference route; its complete-halo MPI schedule has different memory costs.
`mff-export-aoti` and `prepare_for_export` remain compatibility code and are
not supported K100 deployment recipes.

## 5. Run through the public launcher

Expose the matching LAMMPS Python bindings and runtime libraries. Use the
shared library from the same build as those bindings:

```bash
torch_root="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')"
export PYTHONPATH="/path/to/lammps/python:${PYTHONPATH:-}"
mpi_lib=/path/to/the/MPI-used-for-the-build/lib
export LD_LIBRARY_PATH="/path/to/lammps-install/lib:$mpi_lib:$torch_root/lib:$(dirname "$torch_root")/torch.libs:${LD_LIBRARY_PATH:-}"
export HIPBLASLT_ALLOW_TF32=0
```

For the tested Open MPI environment, `mpi_lib="$OPENMPI_HOME/lib"`. Keep this
directory ahead of the Torch wheel libraries. Build or install `mpi4py`
against the same MPI as LAMMPS. The console launcher
initializes MPI before loading Torch and passes that communicator to LAMMPS.
The installed console entry point is the supported launch command.
Keep the system MPI/PMIx selected consistently at runtime. Some vendor Torch
wheels bundle other MPI libraries; if those override the build's MPI, use the
site's approved library ordering or preload the matching MPI/PMIx libraries.
Do not combine a serial LAMMPS library with an MPI launcher.

Example input for an H/C/O potential trained with a 4.5 Å cutoff:

```lammps
units metal
atom_style atomic
atom_modify map array
boundary p p p
read_data system.data

neighbor 0.5 bin
pair_style mff/torch/kk 4.5 cuda
pair_coeff * * /path/to/core.pt H C O

thermo 1
thermo_style custom step pe fmax
run 0
```

Launch inside the batch allocation:

```bash
# One GPU
mff-lammps -in in.mfftorch
# Eight GPUs on one node
mpirun --bind-to none -np 8 mff-lammps -in in.mfftorch
```

The launcher enables Kokkos, full neighbors, Newton off, host communication
for owner maps, GPU-aware MPI off and strict FP32. Kokkos determines device
affinity; the C++ model runtime uses that same device. Visible GPUs are
detected automatically; `--gpus N` overrides the Kokkos GPU count.

For plain graph preparation, use `pair_style mff/torch 4.5 cuda` and
`mff-lammps --plain -in in.mfftorch`. Kokkos is the performance default.
LAMMPS options such as `-var`, `-log` and `-screen` are forwarded by the launcher.

`run 0` evaluates the geometry. Validate energies and complete force vectors
before choosing the actual integrator and timestep. Loading, native kernel
JIT and the first real graph are cold-start costs and belong outside
steady-state timing. Native cores skip synthetic warmup graphs.

## 6. Native execution and ownership

The C++ engine builds edge vectors and differentiates the sum of owned
atomic energies. Single-rank execution folds periodic image ghosts into
local owners. Under MPI, native cores use one-cutoff coordinate halos and
owned-center edges. Every force evaluation refreshes ghost owner rank/local
indices through the current LAMMPS communication plan. Sorting, migration,
repeated images and changes in ghost order do not reuse an old map.

Intermediate features and their adjoints use FP32 RCCL. Standard models
exchange hidden states; FoundationHP exchanges projected features after
`linear_up`. Reverse feature contributions accumulate into their owner,
including repeated ghost rows. Coordinate derivatives then return through
LAMMPS reverse force communication. Weights and arithmetic precision are
unchanged between the performance and capacity schedules.

Native MPI currently requires one node, the full launcher communicator,
one MPI rank per GPU and at least one owned atom per rank. Empty domains and
LAMMPS partitions are rejected. The runtime creates its RCCL process group
automatically and the public launcher releases it before MPI shutdown.
`MASTER_ADDR` and `MASTER_PORT` can override single-node rendezvous defaults.

The engine preserves `forward_distributed`, `warmup_on_real_graph` and
`after_force` from native dispatch archives. `after_force` returns idle Torch
allocator blocks when its reserved pool exceeds 85% of device memory, leaving
space for Kokkos and RCCL allocations. This allocator handoff is included in
step timings. The legacy optional `forward_owned` method and ordinary
reference `forward` remain supported.

## 7. Ownership, pressure and optional extensions

The native ownership schedule is described above. The explicit reference
backend retains depth-dependent coordinate halos and owned-energy sums.
Both exporters write message-passing-depth metadata. Set `depth <layers>`
only for older cores without it; a mismatch with exported depth is rejected.
Any model memory limit must account for the actual owned-plus-ghost input
on each rank.

`MFF_ENABLE_VIRIAL=ON` enables optional Kokkos virial accumulation and defaults
to `OFF`. The plain pair style has a separate virial path. A working force
calculation does not establish pressure, per-atom stress or NPT accuracy.
Validate those quantities for the model, periodic cell and decomposition.

Native learned monopole/dipole/quadrupole reciprocal models use the same
native export and `mff/torch/kk` commands, with configuration restored from
the checkpoint. Their owned-source transport, force-response validation and
1/2/4/8-GPU results are described in the
[electrostatics guide](../../docs/ELECTROSTATICS.md#native-lammps-mfftorchkk).

The source also includes signature-dependent options:

| Option | Model requirement |
| --- | --- |
| `field`, `mfield` | A matching polar/axial vector external-field input |
| `field6`, `field9` | A matching symmetric/full rank-two tensor input |
| `fidelity` | A core accepting runtime fidelity IDs |
| `dispersion` | A core and metadata requiring an explicit dispersion graph |
| Long-range / MBD solvers | Matching source heads, boundary conditions and solver metadata |

These extensions and physical tensor outputs are outside the short-range
energy/force validation described here. `compute mff/torch/phys` is not a
validated general output interface for the exported cores in this guide.

## 8. Troubleshooting

| Symptom | Check |
| --- | --- |
| `.pt2` export or loading fails | Use the native MFF `.pt` export; `.pt2` is unsupported on this stack. |
| Missing `torch/torch.h` | Check the LibTorch CMake prefix and package registration. |
| Unknown `mff/torch` pair style | Check `PKG_USER-MFFTORCH`, copied source files and CMake package inclusion. |
| `libtorch.so` missing | Expose the matching Torch `lib` and wheel-library directories. |
| Kokkos/compiler rejects `gfx926` | Use the compatible vendor toolchain and architecture support. |
| MPI/PMIx initialization fails | Check runtime-library consistency; match the system MPI and PMIx used for the build; verify the loaded `liblammps.so`. |
| Wrong absolute energy | Check E0, model/head selection, species mapping and units. |
| Incomplete pressure | Check the selected virial implementation and its separate numerical validation. |

For force comparisons, inspect complete vectors on the same geometry, cell
and periodic graph. LAMMPS `fmax` is the largest absolute force component,
not the largest per-atom force norm.

## 9. Numerical verification

The K100 checks use DTK 26.04, PyTorch 2.7.1/HIP 6.3, FP32 and TF32 off.
Energy and complete force vectors are compared with the original MACE model
on the same periodic geometry, at `atol=rtol=2e-5`.

- Native C64/L2, two-interaction, correlation-3 H/C/O: 512 physical atoms,
  1/2/4/8 MPI ranks, before and after short NVE integration.
- Native MACE-MH-1, `omat_pbe`: 256 physical Cu atoms, 1/2/4/8 ranks,
  projected-feature exchange, before and after short NVE integration.
- C17/L1, three interactions, correlation 2 and radial hidden widths `[17,31]`:
  512 atoms on 1 and 8 ranks in both performance and capacity modes.
  Nonzero E0, atom sorting and cross-domain migration are included.
- C64 and MH-1 on 8 ranks: fully isolated atoms and mixed connectivity with
  one zero-neighbor domain. These checks retain isolated-atom energies and
  verify reverse communication when a local energy has no physical dependence
  on neighbor features. The native graph retains an exact zero dependency to
  keep the reverse collectives ordered on every rank.
- The portable reference route is checked separately on 64- and 128-atom
  C32/L2/correlation-3 periodic H/C/O graphs, with nonzero E0 and both C++ pair
  styles on 1/2/4/8 ranks. It also checks that an archive exported on one GPU
  can run on another GPU without retaining a fixed tracing device.

### Packaged MFF default: complete MD steps

The current defaults were remeasured on 1/2/4/8 K100 GPUs on 2026-10-22.
See the [README measurements](../../README.md#multi-gpu-results) for current
latency, throughput, strong/weak scaling and large-system demonstrations.
The measurements include the original C64 short-range checkpoint, its
monopole/dipole/quadrupole extensions and pretrained MACE-MH-1 without an
added long-range head. Strong scaling uses 262,144 physical atoms for C64
and 65,536 for MH-1; weak scaling uses 131,072 and 32,768 atoms/GPU.

The native export and C++ MFF/Kokkos pair style run in eager FP32 with TF32
off and two CPU threads per rank. Three warmup steps precede three consecutive
20-step windows; the median window mean is reported. Neighbor rebuilding,
NVE integration, energy, forces, communication and allocator handoff are
included. Each strong- or weak-scaling series uses one exclusive eight-K100
node across all GPU counts. Consecutive windows are not independent runs.
Physical atoms determine throughput; ghost copies are excluded.

For long-range physical conventions and the additional dipole FP64 audit,
see the [electrostatics qualification](../../docs/ELECTROSTATICS.md#current-default-interface-qualification).

The installed `mff-lammps` command is separately checked on 1 and 8 ranks
with Kokkos and plain graph preparation. Eight-rank Kokkos execution also
passes without custom MPI preloading when the matching MPI library directory
precedes Torch wheel libraries.

These checks establish the stated short-range energy/force and communication
coverage. Pressure, NPT and long-trajectory stability are separate validation
questions. Cold initialization/JIT is excluded from steady-state timings;
neighbor rebuilding, integration, force evaluation and communication remain
inside the timed MD step.
