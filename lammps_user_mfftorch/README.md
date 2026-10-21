# LAMMPS USER-MFFTORCH

`USER-MFFTORCH` is a LAMMPS package for running exported MACE-ICTC models through
LibTorch/AOTInductor from C++. It is intended for molecular dynamics (MD) after a model has
already been trained or converted in Python and exported as a deployment core.

## What It Provides

Pair styles:

- `pair_style mff/torch`: C++ + LibTorch/AOTI inference path. Use this first for
  build validation and numerical smoke tests.
- `pair_style mff/torch/kk`: Kokkos/CUDA or HIP data-preparation path plus LibTorch/AOTI
  inference. Use this for GPU MD when the Kokkos build is available.

For supported export/runtime stacks, the recommended model artifact is an
AOTInductor (AOTI) `.pt2` core exported by
`mff-export-aoti` or `python -m mace_ictc.cli.export_aoti_core`. Legacy
TorchScript `core.pt` support is still kept for older LibTorch deployments, but
validation of that deployment route uses `.pt2`. K100 has additional backend
limitations described under [Build](#build).

Notation: N is the number of input nodes, including ghosts when present; E is
the directed edge count; L is the maximum irrep degree. TP means tensor
product, CG means Clebsch–Gordan, and U denotes fixed higher-order coupling
tensors. See the [API terminology](../docs/HP_API.md#terminology-and-notation).

## Generated inference mode

Export defaults to `--operator-plan auto`.
Supported frozen FP32 MACE-ICTC models with positive channels, L<=5,
correlation=1/2/3 and full CG
bridge-U products automatically use generated operators. Unsupported models
keep their existing route and print a fallback reason. Export using:

```bash
python -m mace_ictc.cli.export_aoti_core --checkpoint model.pth --elements H,C,O \
  --angular-basis ictd --embed-e0 \
  --atoms 128 --degree 50 --device cuda --out core_generated.pt2
```

Use `--operator-plan generated` to require generated coverage, or
`--operator-plan checkpoint` to retain the legacy export route. The package
sidecar records the selected plan. This policy does not limit interaction depth
to two layers. Validate the model and deployment backend used for a trajectory.

Use the trained species, dtype and neighbor normalization. `generated` requires
an FP32 checkpoint, no interaction attention and no cuEquivariance (cuEq) modules. TP and angular
harmonics retain the Cartesian ICTC basis; symmetric contraction retains its
existing internal basis bridge. The standalone C++ LibTorch/AOTI runtime was
validated with PyTorch 2.7.1 built for CUDA 12.8; for that route, Python and
Triton are needed during export only. This does not establish K100 AOTI support.

Generated export enables the owned-energy input automatically and writes the
interaction depth into `core_generated.pt2.meta`. Keep the sidecar with the
package. Dynamic N/E export is the default, avoiding a maximum-node padding
allocation when the ghost count varies.

High-L generated export uses bounded sparse correlation=3 contractions and input-degree
TP partitions with channel tiles. Canonical high-L/correlation=3 reference U remains on
CPU after preparation; the package captures only its required factorized GPU
constants. `--reference-buffer-policy device` retains the original U on GPU
during preparation for tooling or memory controls. Checkpoint keys and weights
are retained. Keep the newly exported `.pt2` together with its `.pt2.meta`;
updating Python does not update an existing deployment binary. This selection
does not establish performance or trajectory accuracy for every architecture.

Python checkpoint inference also defaults to auto selection through
`LAMMPS_MLIAP_MFF.from_checkpoint`; pass `operator_plan="checkpoint"` to opt out.
TorchScript keeps the original path. Training auto selection preserves all
original trainable weights and only chooses the full generated training route
when TP base coefficients are already fixed; it never freezes them to obtain
a speedup.

Run the CUDA/Kokkos executable with `-k on g 1 -sf kk -pk kokkos neigh full
newton off comm no`. In the input, use `atom_modify map array` before reading the
data, `pair_style mff/torch/kk <trained-cutoff> cuda`, and the usual element
mapping in `pair_coeff * * core_generated.pt2 ...`. On Slurm clusters, builds,
export, numerical validation and MD must run inside a job submitted with `sbatch`.

Single-rank periodic graphs fold ghosts into their local owners. MPI uses the
exported depth for its halo, selects only the required graph centers, computes
forces from local energies and reverses ghost derivatives even with Newton off.
Older AOTI packages without an ownership input are rejected on an unfolded MPI
graph; re-export them with `--owned-energy-mask` or `--operator-plan generated`.
For older artifacts without depth metadata, set `depth <layers>` in pair_style.

LibTorch follows the actual Kokkos GPU device. Bind GPUs per rank with the scheduler/MPI launcher;
an inherited `SLURM_LOCALID=0` in every Intel-Hydra child is not a valid two-GPU
binding. Energy/force, virial/pressure and trained-potential trajectory accuracy
require their own validation for the user's model and deployment.

Current supported deployment output is energy and force. Physical tensor outputs
are not a supported public LAMMPS interface in the current `.pt2` deployment
path; do not rely on `compute mff/torch/phys` for production runs.

## Install Into LAMMPS

Copy the current package files into the LAMMPS source tree:

- `lammps_user_mfftorch/src/USER-MFFTORCH/` to `LAMMPS/src/USER-MFFTORCH/`
- `lammps_user_mfftorch/cmake/Modules/Packages/USER-MFFTORCH.cmake` to
  `LAMMPS/cmake/Modules/Packages/USER-MFFTORCH.cmake`

Some LAMMPS versions also require adding `USER-MFFTORCH` to package lists in
`LAMMPS/cmake/CMakeLists.txt`. The [build and run guide](docs/BUILD_AND_RUN.md)
covers these edits, backend options and terminology.

## Build

Example CUDA/Kokkos build for A100/A30:

```bash
cd /path/to/lammps
export LIBTORCH_PREFIX="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"

cmake -S cmake -B build-mfftorch-kk \
  -D CMAKE_PREFIX_PATH="$LIBTORCH_PREFIX" \
  -D PKG_USER-MFFTORCH=ON \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_AMPERE80=ON

cmake --build build-mfftorch-kk -j
```

For a non-Kokkos validation build, drop the Kokkos options and use a separate
build directory, for example `build-mfftorch`.

The K100 branch also provides `-D MFF_USE_HIP=ON` against HIP-enabled LibTorch.
For `mff/torch/kk`, additionally enable `PKG_KOKKOS` and `Kokkos_ENABLE_HIP`.
The vendor Kokkos/compiler stack must support gfx926;
graph construction, edge sorting and force writes stay on device. HIP graph
replay is not enabled by this option. Compiling the engine does not make a CUDA `.pt2` package
portable: export/runtime support must be checked on the target stack.
On the tested K100/DTK 26.04 stack, portable TorchScript uses the reference
operators; the optimized eager HP path currently requires an experimental
in-process Python dispatcher bridge. It is not a standalone exported model.
The MPI path disables GPU-aware MPI and uses host communication for ghost
forces. Its single-rank graph folds periodic images; multiple ranks use
depth-dependent halos and local-energy derivatives followed by reverse force
communication. Constant-pressure MD (NPT), virial accuracy and execution across
multiple nodes require separate validation.

For short-range MPI inference, an eager TorchScript bridge can optionally export
`forward_owned(pos, A, batch, src, dst, shifts, cell, edge_vec, nlocal: int)`.

The engine preserves `forward_owned` during freezing and selects it when local and
halo atoms coexist. It must return one energy slot per input atom, with correct
energies for the owned prefix and coordinate derivatives for every required
local/ghost atom. Ghost energy values are unspecified. Ordinary cores continue
to use `forward`.

`PureCartesianICTDFix.forward(..., local_atom_count=nlocal)` provides a matching
short-range inference path: it traces the owned energies' layer dependencies
on the GPU, prunes unnecessary receiver edges and product rows, and retains
all required coordinate gradients. `compact_local_states=True` also compacts
intermediate features and remaps receiver/source indices between layers.
`compact_local_messages=True` additionally writes generated TP messages directly
into the required receiver rows, reducing message allocation and subsequent
projection/self-connection matrix multiplications. It retains gradients with
respect to all required source coordinates.
Both options default to false. This short-range path uses eager execution
through the Python bridge and requires no attention, long-range/dispersion
modules or feature callbacks. It does not reduce the communication halo depth
or create a standalone exported model.

An eager shape-tuned core may also export `warmup_on_real_graph() -> bool`.
The engine preserves this method when freezing. Returning true skips only the
synthetic warmup graph; the first real compute still runs normal energy/force
evaluation. Callers must exclude real-graph initialization/autotuning from
steady-state timing. Existing cores without this method retain their warmup.

At runtime, make sure the PyTorch shared libraries are visible:

```bash
export LD_LIBRARY_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"
```

## Export a Model

Basic `.pt2` export from a MACE-ICTC checkpoint:

```bash
mff-export-aoti \
  --checkpoint model.pth \
  --elements H,C,N,O \
  --out model.pt2 \
  --dynamic \
  --dtype float32 \
  --device cuda \
  --embed-e0
```

For deliberate fixed-N deployment, an example with dynamic edge count is:

```bash
mff-export-aoti \
  --checkpoint model.pth \
  --elements H,C,N,O \
  --atoms 256 \
  --degree 32 \
  --static-n \
  --dynamic \
  --dtype float32 \
  --device cuda \
  --embed-e0 \
  --out model_static256.pt2
```

Rules that matter:

- `--elements` selects checkpoint species. Each `pair_coeff` symbol maps one
  LAMMPS atom type to an exported element.
- `--embed-e0` makes LAMMPS `pe` include the checkpoint's per-element
  atomic-energy offsets (E0).
- `--degree` is the synthetic export graph's average directed degree, not a
  maximum physical coordination number. The example has `E=atoms*degree`.
- `--static-n` fixes N; without `--dynamic`, E is fixed too. Capacity applies
  to each rank's model input, including ghosts. Keep the exported shape
  constraints and `.meta` sidecar consistent with the deployment.
- A loaded checkpoint retains its actual dtype; `--dtype float32` does not
  convert an FP64 checkpoint to FP32.
- Always check the exporter's eager-vs-compiled energy/force comparison before
  running LAMMPS.

## LAMMPS Input

Minimal example:

```lammps
units metal
atom_style atomic
atom_modify map array
boundary p p p

read_data system.data
neighbor 1.0 bin

pair_style mff/torch/kk 5.0 cuda
pair_coeff * * /path/to/model.pt2 H C N O

velocity all create 300 42
fix 1 all nve
run 100
```

Notes:

- `5.0` is the trained cutoff in ångströms (Å) for this example with `metal` units.
- The `pair_coeff` element list maps LAMMPS atom types to element symbols. Use
  `NULL` for atom types that should be skipped.
- Both `mff/torch` and `mff/torch/kk` can load `.pt2`; the `/kk` variant uses
  Kokkos for the GPU data path.

## Multi-GPU MPI Runs

LAMMPS multi-GPU execution uses MPI domain decomposition: run one MPI rank per GPU.
Do not expect one LAMMPS rank to drive all GPUs.

Plain LibTorch/AOTI path:

```bash
export MFF_DEBUG_BUNDLE=1
mpirun -np 2 /path/to/lmp -in in.mfftorch
```

with:

```lammps
pair_style mff/torch 5.0 cuda
pair_coeff * * /path/to/model.pt2 H C N O
```

Kokkos GPU data path:

```bash
export MFF_DEBUG_BUNDLE=1
mpirun -np 2 /path/to/lmp -k on g 2 -sf kk -pk kokkos newton off neigh full comm no -in in.mfftorch
```

with either:

```lammps
pair_style mff/torch/kk 5.0 cuda
pair_coeff * * /path/to/model.pt2 H C N O
```

or `pair_style mff/torch` when `-sf kk` mapping is available in the build.

The plain `mff/torch` engine selects the local GPU device from local-rank
environment variables such as `SLURM_LOCALID`, `LOCAL_RANK`,
`OMPI_COMM_WORLD_LOCAL_RANK`, or `MPI_LOCALRANKID`. The Kokkos variant maps
ranks to Kokkos GPUs correctly for the common single-node, one-rank-per-GPU
case. Validate multi-node Kokkos runs before production.

With `MFF_DEBUG_BUNDLE=1`, initialization prints the requested and selected
device. Check that different local ranks select different GPUs. Before a long
MD run, compare `run 0` energy and forces for `-np 1` and the intended MPI rank count. Small FP32
roundoff differences are possible; large differences usually mean a domain
decomposition, cutoff, ghost, or static `.pt2` capacity issue.

## Numerical comparison

Compare the original checkpoint, converted model, compiled core and LAMMPS
energy/forces on the same geometry and energy convention. LAMMPS `fmax` is the
maximum absolute force component; it differs from the maximum per-atom force
norm. See the [build and run guide](docs/BUILD_AND_RUN.md#8-troubleshooting)
for shape, ownership and pressure checks.
