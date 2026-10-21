# Build and run LAMMPS with USER-MFFTORCH

`USER-MFFTORCH` runs exported MACE-ICTC potentials through LibTorch, the C++
interface to PyTorch. The `mff/torch/kk` pair style additionally uses Kokkos for
GPU graph preparation and force writes. Molecular dynamics (MD) uses the
checkpoint's trained cutoff, species, normalization and energy convention.

This guide covers package installation, model export and execution. For the
Python operators and model configuration, see the [HP API](../../docs/HP_API.md).
For deployment limitations and optional eager interfaces, see the
[package overview](../README.md).

## 1. Requirements and execution environment

Use Linux with:

- A C++17 compiler and CMake compatible with the selected LAMMPS, LibTorch and
  GPU toolchain versions.
- Python 3.10 or later and PyTorch 2.7 or later for this Python package.
- LibTorch matching the export/runtime stack, including its C++ ABI.
- A matching CUDA toolchain for NVIDIA GPUs, or the vendor HIP/DTK toolchain
  and HIP-enabled PyTorch for K100.
- MPI for domain decomposition across multiple processes; Kokkos built for the
  target GPU backend when using `mff/torch/kk`.

Install this repository into an environment that already has the appropriate
vendor PyTorch and Triton packages:

```bash
python -m pip install -e '.[hp,conversion]'
```

Do not replace the K100 vendor stack with a generic CUDA PyTorch installation.
In HIP-enabled PyTorch, `device="cuda"` and `torch.cuda` remain the GPU API
names; see [PyTorch HIP semantics](https://docs.pytorch.org/docs/2.7/notes/hip.html).

On Slurm clusters, submit builds, export/conversion, validation and MD through
`sbatch`. The computation commands below belong inside an allocated batch job;
use `srun` only within that job and follow the site's GPU binding policy.
Source checkout, file copying and editing can be done before submission.

## 2. Obtain LAMMPS

Choose a LAMMPS release compatible with your toolchain. For example, to check
out the [22 July 2025 release](https://github.com/lammps/lammps/releases/tag/stable_22Jul2025):

```bash
git clone https://github.com/lammps/lammps.git
cd lammps
git checkout stable_22Jul2025
```

Keep the LAMMPS and model-export environments consistent. A `.pt2` package
contains compiled code for its export backend; copying it does not convert
CUDA kernels to HIP kernels.

## 3. Install USER-MFFTORCH in the LAMMPS source tree

From the ICTC-O3-HP-K100 repository root:

```bash
lammps_source=/path/to/lammps
mkdir -p "$lammps_source/src/USER-MFFTORCH"
cp -R lammps_user_mfftorch/src/USER-MFFTORCH/. "$lammps_source/src/USER-MFFTORCH/"
```

Copy the package CMake file to the layout used by that LAMMPS release.
For releases using `cmake/Modules/Packages`:

```bash
cp lammps_user_mfftorch/cmake/Modules/Packages/USER-MFFTORCH.cmake \
  "$lammps_source/cmake/Modules/Packages/"
```

For releases using `cmake/Packages` instead:

```bash
cp lammps_user_mfftorch/cmake/Packages/USER-MFFTORCH.cmake \
  "$lammps_source/cmake/Packages/"
```

In `LAMMPS/cmake/CMakeLists.txt`, add `USER-MFFTORCH` to the
`STANDARD_PACKAGES` list. Also add it to the existing `PKG_WITH_INCL` loop
that includes external-library package definitions. If that loop is absent,
add one conditional include, using the same layout as the copied file:

```cmake
if(PKG_USER-MFFTORCH)
  include(Modules/Packages/USER-MFFTORCH)
endif()
```

For the other layout, use `include(Packages/USER-MFFTORCH)` instead. Include the
package definition only once.

## 4. Configure and build

Use the LibTorch CMake prefix from the environment used for deployment:

```bash
export LIBTORCH_PREFIX="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
```

### NVIDIA CUDA build

Select the architecture for the actual GPU. The
[Kokkos architecture table](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html)
lists the options supported by each Kokkos release. Common examples are:

| GPU | Kokkos CMake option |
| --- | --- |
| NVIDIA A100 or A30 | `Kokkos_ARCH_AMPERE80=ON` |
| NVIDIA A10 or A40 | `Kokkos_ARCH_AMPERE86=ON` |
| NVIDIA V100 | `Kokkos_ARCH_VOLTA70=ON` |
| NVIDIA T4 | `Kokkos_ARCH_TURING75=ON` |

For an A100/A30 Kokkos build:

```bash
cd /path/to/lammps
cmake -S cmake -B build-mfftorch-kk \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_PREFIX_PATH="$LIBTORCH_PREFIX" \
  -D PKG_USER-MFFTORCH=ON \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_AMPERE80=ON
cmake --build build-mfftorch-kk --parallel 8
```

For a plain LibTorch build, use a separate build directory and omit the
Kokkos options. Match the number of build workers to the allocated CPU cores.

### K100 HIP build

The package's `MFF_USE_HIP` option links HIP-enabled LibTorch and the HIP host
runtime. The corresponding Kokkos options are:

```bash
cmake -S cmake -B build-mfftorch-hip \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_PREFIX_PATH="$LIBTORCH_PREFIX" \
  -D PKG_USER-MFFTORCH=ON \
  -D MFF_USE_HIP=ON \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_HIP=ON
cmake --build build-mfftorch-hip --parallel 8
```

Add the compiler and target-architecture options required by the vendor
Kokkos/DTK stack. K100 `gfx926` support must exist in that toolchain;
`MFF_USE_HIP=ON` does not add a missing Kokkos architecture definition.

On the K100/DTK 26.04 stack, the portable TorchScript route uses reference
operators. Optimized eager HP execution uses an experimental in-process Python
bridge and is not a standalone AOTInductor model. Building the HIP engine alone
does not establish `.pt2` export/runtime support on that stack. See the
[package overview](../README.md#build) for this distinction.

### Virial, pressure and stress

`MFF_ENABLE_VIRIAL` defaults to `OFF` and controls the Kokkos pair style's
optional virial accumulation. To enable that implementation, configure with
`-D MFF_ENABLE_VIRIAL=ON` and rebuild. The plain `mff/torch` implementation has
its own virial path; this option is not a global switch for both pair styles.

The virial is the configurational contribution used in pressure calculation.
LAMMPS pressure also includes a kinetic contribution. `press` is scalar
pressure, and `pxx pyy pzz pxy pxz pyz` are pressure-tensor components.
They should not be labeled interchangeably as virial or per-atom stress.
See [LAMMPS pressure](https://docs.lammps.org/compute_pressure.html).

With the Kokkos virial path disabled, its missing configurational contribution
makes pressure unsuitable for pressure control. Enabling the option does not
by itself validate pressure, cell derivatives or constant-pressure MD for a
particular model and MPI decomposition. Its overhead depends on the workload.
The LAMMPS [`stress/atom` convention](https://docs.lammps.org/compute_stress_atom.html)
uses the opposite sign and pressure-times-volume units; it is not the same
quantity as the `pxx` thermo field.

## 5. Export a model

### AOTInductor `.pt2`

AOTI means AOTInductor, the ahead-of-time deployment backend of TorchInductor.
The export CLI packages energy/force evaluation and writes a metadata sidecar:

```bash
mff-export-aoti \
  --checkpoint /path/to/model.pth \
  --elements H,O \
  --operator-plan generated \
  --generated-layout packed \
  --device cuda \
  --embed-e0 \
  --out model.pt2
```

Use `generated` to require HP coverage, `auto` to permit a reference fallback,
or `checkpoint` to retain the checkpoint's execution route. Generated export
requires a supported frozen FP32 model. The CLI preserves a loaded checkpoint's
actual dtype; `--dtype float32` does not cast an FP64 checkpoint.

Important export conventions:

- `--elements` selects checkpoint species. Every element mapped by LAMMPS must
  be present in the exported model. The `pair_coeff` symbols map atom types,
  one symbol per LAMMPS atom type.
- `--embed-e0` includes the checkpoint's per-element atomic-energy offsets
  (E0). Use the same energy convention when comparing against Python.
- Dynamic node and edge counts are the default. Node count includes ghost rows
  in an unfolded MPI graph, and can change even when the global atom count is
  fixed. Dynamic dimensions still have the exported program's shape constraints.
- `--atoms` sets the example node count. `--degree` sets its average directed
  degree: the synthetic export graph has `E = atoms * degree`. It is **not** a
  physical coordination limit or a universal maximum neighbor count.
- `--static-n` fixes N to `--atoms`; without `--dynamic`, the example edge count
  is fixed too. `--static-n --dynamic` keeps N fixed and permits dynamic E.
  Check the exported shapes and metadata, including per-rank owned-plus-ghost
  capacity, before using a fixed-N package.
- Keep `model.pt2.meta` next to `model.pt2`. Metadata records ownership support,
  message-passing depth and dynamic-node/padding settings. Retain any additional
  exporter-generated metadata for long-range models as well.
- Check the exporter's eager-versus-compiled energy/force comparisons on the
  target backend before running LAMMPS. Changing Python settings does not update
  a previously compiled package; export again to change kernels.

A deliberately fixed-N export with dynamic edges can use:

```bash
mff-export-aoti \
  --checkpoint /path/to/model.pth --elements H,O \
  --atoms 256 --degree 32 --static-n --dynamic \
  --operator-plan generated --device cuda --embed-e0 \
  --out model_static256.pt2
```

### Convert an upstream MACE checkpoint

For compatible MACE architectures, `mff-convert-mace` converts the original
`ScaleShiftMACE` model to a MACE-ICTC checkpoint:

```bash
mff-convert-mace \
  --mace-model /path/to/mace.model \
  --out mace_ictc_f32.pth \
  --product-backend ictd-bridge-u \
  --dtype float32 --device cpu
```

An FP64 conversion can be retained separately for a precision comparison.
Older MACE pickle files may need matching historical `mace-torch` and e3nn
versions to load. The nonlinear MACE-MH-1 adapter is a separate interface;
follow [Nonlinear MACE foundation checkpoints](../../docs/HP_API.md#nonlinear-mace-foundation-checkpoints)
for that architecture. Its eager inference model is not a drop-in `.pt2`
checkpoint for the command above.

### Legacy TorchScript `.pt`

For a deployment that explicitly needs the legacy TorchScript interface:

```bash
mff-export-core \
  --checkpoint /path/to/model.pth --elements H O \
  --device cuda --dtype float32 --out core.pt
```

Unlike `mff-export-aoti`, this CLI takes a space-separated `--elements` list.
Export and runtime must agree on the core's input signature and backend.

## 6. Run LAMMPS

Expose the matching PyTorch shared libraries at runtime:

```bash
export LD_LIBRARY_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"
```

### Minimal energy/force input

Prepare `system.data` with a meaningful geometry, masses and atom-type mapping.
This example assumes an H/O model trained with a 5 Å cutoff, `metal` units,
and a compatible `.pt2` runtime:

```lammps
units metal
atom_style atomic
atom_modify map array
boundary p p p
read_data system.data

neighbor 1.0 bin
pair_style mff/torch 5.0 cuda
pair_coeff * * /path/to/model.pt2 H O

thermo 1
thermo_style custom step pe fmax
run 0
```

For Kokkos, use `pair_style mff/torch/kk 5.0 cuda`. A single-GPU launch inside
an allocated batch job is:

```bash
/path/to/lmp -k on g 1 -sf kk -pk kokkos newton off neigh full comm no -in in.mfftorch
```

Here, `neigh full` requests a full neighbor list, and `newton off` selects the
supported Newton setting for this interface. After checking energy and forces,
choose velocities, timestep and integration settings appropriate to the
potential and system. For example, microcanonical (NVE) integration uses `fix ... nve`;
`run 0` above evaluates a configuration without advancing MD.

Current public deployment output is energy and forces. Physical tensor outputs
are not supported by the public `.pt2` interface; `compute mff/torch/phys` is
not a supported production output interface.

### MPI domain decomposition

Use one MPI rank per GPU. MPI partitions physical atoms into owned atoms and
communicates ghost copies near subdomain boundaries. The interface evaluates
owned energies and returns force contributions on ghost coordinates to their
owners through reverse communication.

For example, inside a batch allocation with two GPUs visible to Kokkos:

```bash
export MFF_DEBUG_BUNDLE=1
mpirun -np 2 /path/to/lmp \
  -k on g 2 -sf kk -pk kokkos newton off neigh full comm no -in in.mfftorch
```

For the plain pair style, omit the Kokkos command-line options. GPU visibility
and local-rank mapping must match the scheduler/MPI launch configuration.
When each rank sees only one GPU, use the corresponding one-GPU Kokkos setup.
`MFF_DEBUG_BUNDLE=1` prints device selection; verify distinct devices for distinct
local ranks. An inherited `SLURM_LOCALID=0` in every child of an Intel MPI Hydra
launcher does not provide valid per-rank binding.

Single-rank periodic execution folds image ghosts into local owners. Multi-rank
execution uses depth-dependent halos. The sidecar's `message_passing_depth`
must match the model's interaction depth; for an older artifact without it,
set `depth <layers>` in `pair_style`. An unfolded MPI graph also needs an
ownership-aware exported core. Re-export older packages with
`--owned-energy-mask` or `--operator-plan generated` when that input is absent.

For fixed-N packages, capacity applies to each rank's model input, including
its ghosts. A fixed global atom count does not fix those input sizes.
Compare single-rank and multi-rank energy and forces for the same geometry
before a trajectory. Different FP32 reduction orders can cause small numerical
differences. Large discrepancies require checking ownership, halo depth, cutoff,
neighbor filtering, device mapping and exported shape constraints.

### Pressure output

With a validated virial path, pressure can be requested as:

```lammps
thermo_style custom step temp pe ke etotal press pxx pyy pzz pxy pxz pyz
```

See [Virial, pressure and stress](#virial-pressure-and-stress) for the build
option and definitions. Energy/force agreement alone does not establish
pressure accuracy or constant-pressure ensemble support.

## 7. Model-specific optional inputs

These inputs require an architecture and exported signature that explicitly
support them. They are not enabled by a short-range HP operator plan.

### External fields

The legacy TorchScript interface can accept a vector external field through
LAMMPS equal-style variables:

```lammps
variable Ex equal 0.0
variable Ey equal 0.0
variable Ez equal 0.01
pair_style mff/torch 5.0 cuda field v_Ex v_Ey v_Ez
pair_coeff * * /path/to/core.pt H O
```

`field` supplies a polar vector (`1o`); `mfield` supplies an axial vector
(`1e`) when required by the model. These are tensor ranks, distinct from MPI
process ranks. Variables are evaluated at each force call. A model requiring
an external field cannot omit it, and a core without that input cannot accept
these options.

A supported rank-two tensor input uses `field9` in row-major order
`xx xy xz yx yy yz zx zy zz`, or `field6` for the symmetric tensor entries
`xx yy zz xy xz yz`. Supply nine or six equal-style variable references,
respectively. Do not combine these rank-two options with vector-field options.

### Many-body dispersion

MBD means many-body dispersion. For models with a separate real-space
dispersion graph, supply its trained cutoff independently of the short-range
message-passing cutoff:

```lammps
pair_style mff/torch 4.0 cuda dispersion 6.0
pair_coeff * * /path/to/core.pt H C N O
```

The engine requests candidate neighbors out to the larger cutoff and constructs
separate edge lists. The dispersion cutoff must equal the checkpoint/export
metadata; changing it changes the coupling graph. A graph with no dispersion
edges is different from omitting a required dispersion input.

`mbd` uses a dense eigensolver as a reference. `mbd-slq` denotes the approximate
MBD route based on stochastic Lanczos quadrature (SLQ); probe count and
Lanczos steps control its approximation. For AOTI export of the `edge_sparse`
backend, the exporter selects `atom-rademacher` probes and the
`newton-schulz` matrix-square-root approximation instead of eigendecomposition,
and writes `dispersion_edges 1` metadata. This changes the numerical
approximation and requires comparison with the intended reference settings.
The `pme_fft` backend is a separate reciprocal-space particle-mesh Ewald (PME)
operator and does not use this real-space dispersion edge contract. Dense
`mbd` is not supported for AOTI export.

Long-range and dispersion routes require their own numerical and backend
validation; the short-range K100 capacity results do not characterize them.

## 8. Troubleshooting

| Symptom | Checks |
| --- | --- |
| `torch/torch.h` missing | Check `CMAKE_PREFIX_PATH`, the copied package file and its conditional include in LAMMPS CMake. |
| `Unrecognized pair style 'mff/torch'` | Check `PKG_USER-MFFTORCH=ON`, package-list registration and copied source files. |
| `libtorch.so` missing at runtime | Set `LD_LIBRARY_PATH` to the libraries from the matching PyTorch environment. |
| Unsupported GPU architecture | Use a Kokkos/compiler configuration supporting the actual device; K100 requires vendor `gfx926` support. |
| Full-neighbor/Newton error | Launch the Kokkos path with `-pk kokkos newton off neigh full comm no`. |
| Wrong energy offset | Check E0 inclusion, units, species mapping and the selected checkpoint/head. |
| Wrong force comparison | Compare complete forces for the same geometry, cell and periodic edges. LAMMPS `fmax` is the maximum absolute force component, not the largest per-atom force norm. |
| MPI-only discrepancies | Check per-rank GPU mapping, owned-energy input, message depth, ghost communication and fixed-N capacity. |
| Incomplete Kokkos pressure | Check `MFF_ENABLE_VIRIAL`, rebuild, then validate the model and decomposition's virial path. |

The definition of `fmax` follows [LAMMPS thermo output](https://docs.lammps.org/thermo_style.html).

## 9. Source layout

```text
ICTC-O3-HP-K100/
├── ictc_o3/                       # Standalone operator API
├── mace_ictc/
│   └── cli/
│       ├── convert_mace.py
│       ├── convert_foundation.py
│       ├── export_aoti_core.py
│       └── export_libtorch_core.py
└── lammps_user_mfftorch/
    ├── src/USER-MFFTORCH/
    ├── cmake/Modules/Packages/USER-MFFTORCH.cmake
    ├── cmake/Packages/USER-MFFTORCH.cmake
    └── docs/BUILD_AND_RUN.md
```
