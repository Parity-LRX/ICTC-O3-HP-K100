# LAMMPS USER-MFFTORCH on K100

`USER-MFFTORCH` runs MACE-ICTC potentials in LAMMPS on Hygon K100 (`gfx926`).
The default is native HP inference with Kokkos/HIP graph preparation and
layerwise FP32 RCCL feature exchange. `mff-export-core` packages the model;
`mff-lammps` loads it and selects the runtime automatically.

AOTInductor `.pt2`, whole-model `torch.compile` and graph replay are unsupported
on the current K100 stack. Native HIP kernel JIT compilation is required and
is separate from those whole-model compilation mechanisms.

## Build and install

Follow the [build guide](docs/BUILD_AND_RUN.md) to build shared LAMMPS with
`PKG_USER-MFFTORCH=ON`, `MFF_USE_HIP=ON`, `PKG_KOKKOS=ON` and
`BUILD_MPI=ON`. Install this repository and the matching LAMMPS Python bindings
in the same HIP-enabled Python environment, including `.[hp,conversion,lammps]`
for the MPI Python dependency. Keep MPI, DTK, PyTorch/LibTorch
and the C++ ABI consistent.

On a Slurm cluster, submit compilation, conversion, export and MD with
`sbatch`. Run the following commands inside the allocated K100 job.

## Export once

For a converted standard MACE checkpoint:

```bash
mff-export-core --checkpoint converted.pth --elements H C O --out core.pt
```

The defaults are `--backend native --inference-mode performance`, FP32 and
E0 included. Channel count, angular degrees, interaction count, radial MLP,
cutoff and species come from the checkpoint. Forward message chunks are
bounded using the actual tensor-product width and device memory; backward
chunks are bounded separately. No model dimensions are fixed to the C64 test.

For a converted MACE-MH-1 `FoundationHP` object:

```bash
mff-export-core --checkpoint mace-mh-1-ictc-hp.pt --elements Cu \
  --head omat_pbe --out mh1-core.pt
```

The selected head is used for evaluation; all original heads remain in the
checkpoint. Use species present in the checkpoint and the head appropriate
for the intended calculation.

Keep these four files together when moving an exported model:

- `core.pt`: small TorchScript dispatch archive.
- `core.pt.checkpoint.pth`: weights and architecture.
- `core.pt.json`: runtime configuration, species and head.
- `core.pt.meta`: C++ backend and message-passing metadata.

The native backend calls the installed Python HP runtime from LibTorch.
It reconstructs the model and registers dispatch operators automatically;
users do not need a private adapter or manual operator registration.
A Python checkpoint renamed to `.pt` is not a deployment archive.

Use `--inference-mode capacity` for smaller forward message chunks. This
changes memory scheduling while retaining FP32 arithmetic and model weights.
A separate `--backend reference` option exports the original portable
whole-network TorchScript route. It uses reference operators and a complete
message-depth coordinate halo under MPI; its speed and capacity differ.

## Run directly

Example input for an H/C/O potential trained with cutoff 4.5 Å and `metal` units:

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

After setting the binding and library paths from the build guide, launch
on one GPU or eight GPUs inside the allocation:

```bash
mff-lammps -in in.mfftorch
mpirun --bind-to none -np 8 mff-lammps -in in.mfftorch
```

`mff-lammps` enables Kokkos, full neighbors, Newton off and host communication
for the small owner map. It detects the visible GPU count; use `--gpus N` if
the site's launcher needs an explicit Kokkos device count. `cuda` is PyTorch's
GPU device name on the K100 HIP build. The C++ engine binds HP execution to
the device selected by Kokkos.

For plain graph preparation, use `pair_style mff/torch 4.5 cuda` and
`mff-lammps --plain -in in.mfftorch`. The Kokkos route is the performance default.

`run 0` evaluates the configuration. Set the pair-style cutoff to `max_radius` recorded in `core.pt.json`.
Choose the timestep, velocities and
integrator for the actual potential before running MD. Keep the trained
cutoff, neighbor normalization, units, E0 convention and element mapping.

## Ghosts and communication

Single-rank execution folds periodic image ghosts into their local owners.
For native MPI execution, the coordinate halo covers one cutoff. Every force
evaluation refreshes the ghost owner rank and local index through the current
LAMMPS communication plan. Intermediate features and their reverse gradients
use FP32 RCCL; repeated periodic images accumulate their adjoints into the
same owner. A zero dependency keeps reverse collectives ordered even for
zero-neighbor domains, while preserving isolated-atom energies. Coordinate derivatives are returned through LAMMPS reverse force
communication. Migration and atom sorting do not reuse a stale owner map.

Standard MACE models exchange hidden states between interactions. FoundationHP
models exchange projected features after `linear_up`, retaining only owned
hidden states. Message depth is restored from the export metadata.

The qualified native MPI scope is one node, one MPI rank per GPU, a single
LAMMPS communicator and at least one owned atom per rank. LAMMPS partitions
and empty domains are rejected. Multi-node execution requires separate
qualification and is not enabled by these defaults.

## Conversion and precision

```bash
mff-convert-mace --mace-model original.model --out converted.pth \
  --product-backend ictd-bridge-u --dtype float64 --device cpu
```

The standard converter preserves radial MLP depth and hidden widths, normalized
activations, TP path normalization, E0 and energy scale/shift. An empty hidden
list is a single linear projection. Legacy checkpoints without
`radial_mlp_hidden` retain `[64,64,64]`.

The supported standard architecture uses ordinary/residual MACE interactions,
Bessel radial features, natural-parity contiguous hidden degrees with uniform
multiplicity, uniform correlation, SiLU readout and at least two interactions.
Nonlinear foundation models use `mff-convert-foundation`; see the
[conversion API](../docs/HP_API.md#6-checkpoint-conversion-and-lammps-on-k100).
Native MFF currently exports short-range scalar energy models. Extra input
heads, physical tensor outputs and long-range/dispersion extensions require
their corresponding implementation and validation.

Native inference uses FP32 with TF32 disabled. Export retains the original
checkpoint; the runtime creates the FP32 inference model. Validate energy
and complete force vectors against the source model in the same precision.

## Pressure and validation scope

Energy and forces are the primary outputs. Kokkos virial accumulation is
controlled by `MFF_ENABLE_VIRIAL`, which defaults to `OFF`. Pressure, per-atom
stress, NPT and long-trajectory stability require their own numerical checks.
LAMMPS `fmax` is the largest absolute force component, not an atom's force norm.

The repository's existing scaling figures were measured with dedicated eager
ML-IAP/RCCL adapters. Their original scripts have been rerun on K100; the MFF
native default uses the same native kernels and layerwise transport strategy.
See the [build guide](docs/BUILD_AND_RUN.md#9-numerical-verification) for the
separate checks of the shipped MFF path.
