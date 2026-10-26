# MACE-ICTC user guide

Train an energy-and-force potential, convert a supported MACE model, and run
it with Python or LAMMPS on Hygon K100. The commands below use the public
`mff-*` command-line tools; their Python package is named `mace_ictc`.

## Choose a workflow

| Your starting point | Workflow |
| --- | --- |
| Structures with reference energies and forces | [Prepare data](#2-prepare-training-data), then [train a short-range model](#3-train-without-long-range-electrostatics) |
| Data requiring an explicit reciprocal correction | [Train with learned electrostatics](#4-train-with-long-range-electrostatics) |
| An ordinary supported MACE `ScaleShiftMACE` model | [Convert the model](#6-convert-an-existing-mace-model) |
| A MACE-MH-1 pretrained model | [Use the foundation converter](#7-mace-mh-1-pretrained-models) |
| A trained or converted ICTC checkpoint | [Predict with Python](#8-predict-energy-and-forces-with-python) or [export to LAMMPS](#10-export-and-run-in-lammps) |

The K100 performance workflow is **eager FP32**. Native HIP kernels compile
when first used. Whole-model `torch.compile`, `--train-makefx-compile` and
AOTInductor `.pt2` deployment are outside this K100 recipe.

## 1. Install and prepare a working directory

Use a Linux Python environment containing your site's K100-compatible
PyTorch, DTK/HIP and Triton. Preserve that vendor stack when installing:

```bash
git clone https://github.com/Parity-LRX/ICTC-O3-HP-K100.git
cd ICTC-O3-HP-K100
python -m pip install -e '.[hp,train,conversion,lammps]'
python -m pip install tables
```

Run these installation commands inside an allocated compute job.
`tables` supplies the PyTables backend used by Pandas during preprocessing.
The `conversion` extra installs upstream MACE; it is used for conversion,
not required by the independent HP operators with full Clebsch–Gordan (CG)
coefficients. `lammps` installs
`mpi4py`; the LAMMPS shared library and bindings are built separately below.

Keep datasets, models, logs and compilation caches in a working directory:

```bash
export ICTC_REPO=/path/to/ICTC-O3-HP-K100
export ICTC_ENV=/path/to/k100-python-environment
mkdir -p /path/to/my-potential/{data,models}
cd /path/to/my-potential
```

### Running on Slurm

All preprocessing, conversion, training, prediction, compilation and MD
commands in this guide belong inside an allocated compute job. The supplied
[`k100_job.sbatch`](../examples/k100_job.sbatch) accepts a command and its
arguments. It uses `DCU_K100`, `--gres=dcu:1` and DTK 26.04 as site examples;
edit those settings and supply your own account:

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/k100_job.sbatch" \
  mff-train --help
```

The job starts in the directory from which `sbatch` was submitted. Wait for a
job to complete successfully before submitting the next dependent stage.
Its log is `ictc-JOBID.log`; caches live in `.cache/ictc/` in your working
directory. A one-GPU request is sufficient for the single-GPU recipes.

If your site needs extra compiler or MPI setup, export
`ICTC_SITE_ENV=/absolute/path/to/site_env.sh`. The batch template sources
that file after loading DTK. For example, a site using GCC 11 host libraries
may require the following in that file:

```bash
export HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11"
# Load the same MPI installation for LAMMPS, mpi4py and the MPI launcher.
module load compiler/openmpi/openmpi-4.1.6-amd
```

These are installation-specific settings, not universal compiler paths.
PyTorch still spells its HIP device `cuda`, including in all commands below.

## 2. Prepare training data

Supply an extended XYZ file containing multiple reference configurations:

- Element symbols and Cartesian positions for every atom.
- A **total energy per configuration** and a three-component force per atom.
- Cell vectors and PBC flags. The periodic electrostatics example requires
  three-dimensional periodic cells.

Use consistent units. The examples use Å, eV and eV/Å, matching LAMMPS
`units metal`. For example, labels can be named `REF_energy` in the frame
metadata and `REF_forces` in the per-atom `Properties` fields. These are
reference labels supplied by your electronic-structure calculation.

Place the file at `data/reference.extxyz`, then preprocess it:

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/k100_job.sbatch" \
  mff-preprocess \
    --input-file data/reference.extxyz --output-dir data/hco \
    --elements H C O --atomic-energy-keys 1 6 8 \
    --energy-key REF_energy --force-key REF_forces \
    --max-radius 4.5 --train-ratio 0.9 --seed 42 --num-workers 4
```

Replace the elements and atomic numbers for your chemistry. List every
present atomic number in ascending order in `--atomic-energy-keys`. The prepared
directory contains `processed_train.h5`, `processed_val.h5`, `fitted_E0.csv`
and split-index files. Set the **same neighbor cutoff** when training; the
dataset loader checks it. Do not use `--skip-h5` for this training workflow.

The default split is random. If every frame carries a `source` metadata
field, preprocessing can select a source-tail holdout instead. Inspect the
reported split; use a suitable independent validation set for your application.

### Atomic energy offsets: an explicit step

The training CLI **does not automatically read `fitted_E0.csv`**. Without
explicit E0 arguments it uses existing H/C/N/O fallback values, which need
not match your calculation. Pass `--atomic-energy-keys` and
`--atomic-energy-values` together.

The supplied [`train_hco.sh`](../examples/train_hco.sh) reads the `Atom,E0`
columns of `DATA_DIR/fitted_E0.csv` and supplies both arguments. To use your
own offsets, export `E0_CSV=/path/to/my_E0.csv` before submission. A CSV looks
like this, with energies replaced by the appropriate values for your labels:

```text
Atom,E0
1,-13.6
6,-1000.0
8,-2000.0
```

Those numbers illustrate the file format; they are not recommended atomic
energies. Preprocessing retains total reference energies. The trainer adds
E0 once when constructing total predicted energies. Fitted offsets can be
underdetermined for data with a fixed composition; use your chosen reference
atomic energies when available.

For a large dataset, `--shards N` performs sharded preprocessing. Its merged
H5 files contain external links into `shard_*/`: keep those directories,
retain their paths, and use `HDF5_USE_FILE_LOCKING=FALSE` if required by your
filesystem. Start with the unsharded recipe above.

## 3. Train without long-range electrostatics

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/k100_job.sbatch" \
  bash "$ICTC_REPO/examples/train_hco.sh" short data/hco models/hco-short.pth
```

This recipe explicitly disables electrostatics and dispersion. It trains
C64, lmax=2, two interactions and correlation=3, using a 4.5 Å cutoff,
cutoff polynomial `p=6`, full CG, ICTC angular features, AdamW and an
energy-plus-force MSE loss. The base tensor-product (TP) coefficients are frozen; radial,
linear, self-connection and symmetric-contraction weights remain trainable.

The essential HP options are:

```bash
--dtype float32 --device cuda \
--product-backend ictd-bridge-u --angular-basis ictd \
--freeze-conv-tp-weight --operator-plan generated --generated-layout fused_pack \
--training-message-linear-layout node_component
```

`generated` requires supported HP coverage and reports an error if it cannot
provide it. `auto` permits reference fallback; `reference` explicitly selects
the differentiable reference route. Those switches do not change precision
or freeze parameters automatically. `node_component` is already the training
default and is explicit here for reproducibility.

Extra arguments after the checkpoint override recipe defaults:

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/k100_job.sbatch" \
  bash "$ICTC_REPO/examples/train_hco.sh" short data/hco models/hco-short.pth \
    --channels 128 --lmax 3 --max-ell 3 --correlation 2 --num-interaction 3 \
    --epochs 200 --batch-size 2 --lr 3e-4
```

Channels, degree, correlation and interaction count are configurable; the
generated symmetric-contraction route currently covers correlation 1, 2 and 3.
The model above is a starting example. Reduce batch size first if memory is
insufficient. `--gradient-accumulation-steps 4` accumulates four microbatches
per optimizer update. The effective batch is
`batch_size × GPU_count × gradient_accumulation_steps`.

The checkpoint path stores the best validation weights. Recovery snapshots
can be enabled with `--checkpoint-interval-steps 1000`. Read validation
energy/force errors and checkpoint messages in the job log before deployment.

## 4. Train with long-range electrostatics

Long-range electrostatics is **off by default** in the training CLI. Enable
it while training the appropriate source heads:

| Recipe mode | Learned moments | Required backbone degree |
| --- | --- | --- |
| `short` | None | No multipole requirement |
| `q` | Monopoles | lmax ≥ 0 |
| `qmu` | Monopoles and dipoles | lmax ≥ 1 |
| `qmuq` | Monopoles, dipoles and quadrupoles | lmax ≥ 2 |

For example, train all three moment ranks:

```bash
export ICTC_MESH=32,32,32
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/k100_job.sbatch" \
  bash "$ICTC_REPO/examples/train_hco.sh" qmuq data/hco models/hco-long.pth \
    --batch-size 2 --epochs 200
```

The long-range recipe adds these options to the short-range training setup:

```bash
--long-range-mode reciprocal-spectral-v1 \
--long-range-boundary periodic --long-range-reciprocal-backend mesh_fft \
--long-range-mesh-size 32,32,32 --long-range-source-channels 1 \
--long-range-max-multipole-l 2 \
--long-range-assignment pcs --long-range-green-mode poisson \
--long-range-mesh-fft-full-ewald --long-range-mesh-fft-reciprocal-only \
--long-range-smearing-sigma 1.0 --electrostatics-backend hip
```

The two mesh flags together select a **Gaussian-screened reciprocal-only
correction**. Despite the historical `full-ewald` option name, this recipe
does not add the real-space, self or background terms of a complete Ewald
model. The learned moments are latent model outputs trained through the
energy/force loss; they are not prescribed oxidation states or supervised
physical charges. Neutralization is enabled and the k=0 mode is excluded.
The long-range implementation supports learned monopole, dipole and quadrupole sources.

`32³` is an example grid, not an accuracy setting for every cell. Choose
and check grid resolution for the cell sizes in your dataset. Rectangular
cells can use, for example, `ICTC_MESH=32,32,64`. Sigma is a fixed physical
length in position units. Mesh, assignment, boundary and sigma are saved in
the checkpoint and must remain consistent during evaluation and deployment.
See [physical conventions and grid selection](ELECTROSTATICS.md).

### Add a head to an existing short-range ICTC model

With the same data and E0 convention, warm-start the trained backbone:

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/k100_job.sbatch" \
  bash "$ICTC_REPO/examples/train_hco.sh" qmuq data/hco models/hco-long.pth \
    --resume-checkpoint models/hco-short.pth --finetune --lr 1e-4
```

`--finetune` permits missing keys for the newly added head and starts a fresh
optimizer. Inspect the reported missing/unexpected keys: they should match
the intended extension. Train and validate the head; adding it at inference
time does not turn a short-range checkpoint into a trained long-range model.
Use actual variable-size batches, without node padding or make_fx bucketing,
for these nonlocal models.

## 5. Multiple GPUs, restart and evaluation

### Single-node DDP training on eight K100 GPUs

Run one `torchrun` launcher with eight workers, using the same training
arguments as the single-GPU recipe. For the complete recipe script:

```bash
export ICTC_OMP_THREADS=2
sbatch --account=YOUR_ACCOUNT --ntasks=1 --gres=dcu:8 \
  --cpus-per-task=32 --mem=256G "$ICTC_REPO/examples/k100_job.sbatch" \
  torchrun --standalone --nnodes=1 --nproc-per-node=8 --no-python \
    bash "$ICTC_REPO/examples/train_hco.sh" short data/hco models/hco-ddp.pth \
      --ddp on --batch-size 2
```

Use `q`, `qmu` or `qmuq` for DDP with learned electrostatics. Keep one process
per GPU. `batch-size` is per worker; DDP distributes configurations, whereas
LAMMPS MPI below partitions a single physical system. Do not launch eight
copies of this eight-worker `torchrun` command.
`ICTC_OMP_THREADS` bounds CPU threads per worker; choose data-loader worker
counts to fit the allocation as well.

### Continue an interrupted training run

Supply the **same architecture, E0 and long-range settings** as the original
run and restore training state:

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/k100_job.sbatch" \
  bash "$ICTC_REPO/examples/train_hco.sh" short data/hco models/hco-short.pth \
    --resume-checkpoint models/hco-short.recovery.pth --resume-training-state \
    --checkpoint-interval-steps 1000 --epochs 200
```

The recovery filename is formed from the checkpoint stem. It exists only if
recovery saving was enabled and reached its interval. Weights-only warm-start
uses `--resume-checkpoint` without `--resume-training-state`. Do not combine
`--finetune` with restoring training state.

Optimizer state and the global step are restored. Stateful scheduler history
is not serialized, so those schedulers resume approximately; keep the original
scheduler configuration.

For one validation pass, add `--eval-only --resume-checkpoint CHECKPOINT` to
the matching recipe. This evaluates the prepared validation split.

## 6. Convert an existing MACE model

### Ordinary MACE models

The standard converter expects a **torch-saved `ScaleShiftMACE` module**,
usually a `.model` file. A filename ending in `.ckpt` is fine only if its
payload is that module. A state-dict-only training checkpoint needs the
original MACE architecture reconstructed/exported first; renaming it does
not supply the missing architecture.

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/k100_job.sbatch" \
  mff-convert-mace --mace-model models/source.model \
    --out models/mace-ictc.pth --product-backend ictd-bridge-u \
    --dtype float32 --device cpu
```

Conversion reads the source architecture and transforms learned tensors to
the ICTC-compatible representation. It retains E0, energy scale/shift,
neighbor normalization, cutoff and supported radial-network dimensions.
`float32` prepares this example for native HP inference. Use `--dtype float64`
for a higher-precision conversion/reference checkpoint; the native LAMMPS
exporter creates an FP32 runtime model explicitly.

The standard converter supports ordinary/residual interactions, at least
two interactions, contiguous natural-parity hidden degrees with uniform
multiplicity, Bessel radial features, uniform correlation and SiLU readout.
Extra distance transforms, repulsion and nonlinear foundation interactions
need the foundation route or a dedicated adapter. A reduced-CG checkpoint
also retains its basis; conversion does not silently replace it with full CG.
Current standalone HP coverage is the full-CG bridge-U path.

Run the [Python example](#8-predict-energy-and-forces-with-python) on the same
structures as the original MACE model and compare total energy and complete
force vectors at matching precision before using the converted potential.

### Train further from a converted ordinary checkpoint

The CLI supports compatible standard conversions using its existing radial
MLP `[64,64,64]`. It restores stored channel counts, angular degrees, cutoff and other mapped
architecture settings. Conversion itself supports additional radial widths;
that does not imply that this training CLI can rebuild all such variants.
Use the model/trainer API for architectures outside the CLI builder.

Preserve the source E0 when changing datasets. Extract it in a compute job
to an `Atom,E0` CSV:

```python
import csv
import torch

checkpoint = torch.load("models/mace-ictc.pth", map_location="cpu", weights_only=False)
with open("models/source-E0.csv", "w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["Atom", "E0"])
    writer.writerows(zip(checkpoint["atomic_energy_keys"].tolist(),
                        checkpoint["atomic_energy_values"].tolist()))
```

Save that block as `extract_e0.py` and submit it through the batch template:

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/k100_job.sbatch" python extract_e0.py
```

After that job completes:

```bash
export E0_CSV="$PWD/models/source-E0.csv"
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/k100_job.sbatch" \
  bash "$ICTC_REPO/examples/train_hco.sh" short data/hco models/mace-finetuned.pth \
    --resume-checkpoint models/mace-ictc.pth --lr 1e-4
```

This uses a fresh optimizer and strict weight loading. Keep the preprocessed
cutoff equal to the source checkpoint cutoff. The architecture override does
not rebuild your dataset. For an added long-range head, select its mode and
add `--finetune` as in section 4. Inspect reconstruction and load messages.
Unset `E0_CSV` before returning to recipes that use the dataset's fitted E0.

## 7. MACE-MH-1 pretrained models

MACE-MH-1 uses nonlinear foundation interactions and a separate converter:

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/k100_job.sbatch" \
  mff-convert-foundation --mace-model models/mace-mh-1.model \
    --head omat_pbe --out models/mace-mh-1-ictc-hp.pt
```

All original heads are retained; `--head` selects the default evaluation head.
Choose a head and elements supported by your actual model and application.
The converted object preserves the nonlinear architecture, distance
transform, repulsion and original readouts while using HP tensor-product
and symmetric-contraction operations.
It is not identical to converting every internal feature into the standalone
Cartesian training architecture: the optimized foundation SC also consumes
the checkpoint's existing e3nn-basis features directly.

The current `FoundationHP` adapter supports **frozen-weight energy/force
inference**. It is not an MH-1 fine-tuning trainer. Use the ordinary conversion
workflow only for architectures supported by that converter.

## 8. Predict energy and forces with Python

The supplied [`predict.py`](../examples/predict.py) reads one ASE-readable
structure, builds its neighbors using the checkpoint cutoff, and writes
total energy and forces, including E0 exactly once. Supply a nonsingular cell
even for a nonperiodic molecule. It expects energies in eV and lengths in Å.

```bash
# A trained or converted ordinary FP32 checkpoint.
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/k100_job.sbatch" \
  python "$ICTC_REPO/examples/predict.py" --checkpoint models/mace-ictc.pth \
    --structure data/reference.extxyz --index 0 --out prediction.json

# MACE-MH-1; use a structure whose elements are in this model.
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/k100_job.sbatch" \
  python "$ICTC_REPO/examples/predict.py" --foundation \
    --checkpoint models/mace-mh-1-ictc-hp.pt --head omat_pbe \
    --structure data/cu.extxyz --out mh1-prediction.json
```

The standard example also handles trained reciprocal models: their source
heads and electrostatic functional are restored from the checkpoint. Pass
`--reference` to use reference operators and retain the stored precision.
Force evaluation requires autograd even with
frozen parameters, so this example does not wrap prediction in
`torch.inference_mode()`.

## 9. Compile the LAMMPS interface

### Prerequisites and package installation

Use a LAMMPS/Kokkos source tree with vendor K100 `gfx926` support. The tested
base is `22Jul2025_update6` with that support; an unmodified upstream tree
may not recognize `Kokkos_ARCH_AMD_GFX926`. This repository supplies the
`USER-MFFTORCH` package, not a complete vendor LAMMPS distribution.

Use the same MPI for LAMMPS, `mpi4py` and `mpirun`, and the same HIP PyTorch
environment for compilation and execution. From the repository, copy:

```bash
export LAMMPS_SOURCE=/path/to/vendor-lammps
mkdir -p "$LAMMPS_SOURCE/src/USER-MFFTORCH"
cp -R "$ICTC_REPO/lammps_user_mfftorch/src/USER-MFFTORCH/." \
  "$LAMMPS_SOURCE/src/USER-MFFTORCH/"
cp "$ICTC_REPO/lammps_user_mfftorch/cmake/Modules/Packages/USER-MFFTORCH.cmake" \
  "$LAMMPS_SOURCE/cmake/Modules/Packages/"
```

In `LAMMPS_SOURCE/cmake/CMakeLists.txt`, add `USER-MFFTORCH` to
`STANDARD_PACKAGES` and its existing `PKG_WITH_INCL` package loop. If that
loop is absent, conditionally include the package definition **after the
`lammps` target exists**, once:

```cmake
if(PKG_USER-MFFTORCH)
  include(Modules/Packages/USER-MFFTORCH)
endif()
```

For trees using `cmake/Packages/`, copy this repository's corresponding
`cmake/Packages/USER-MFFTORCH.cmake` and use `include(Packages/USER-MFFTORCH)`.
The full [build guide](../lammps_user_mfftorch/docs/BUILD_AND_RUN.md) covers
both layouts and vendor-library path adjustments.

### Configure, build and install inside a batch job

Save this as `build_lammps.sh`. Set `LAMMPS_SOURCE` above and export
`LAMMPS_INSTALL=/path/to/my-lammps-install`. Use a fresh build directory for
this MPI/HIP configuration:

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${LAMMPS_SOURCE:?Set the vendor LAMMPS source directory}"
: "${LAMMPS_INSTALL:?Set the installation prefix}"
torch_root="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')"
cmake -S "$LAMMPS_SOURCE/cmake" -B build-lammps-ictc \
  -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_STANDARD=17 \
  -D CMAKE_CXX_COMPILER="$(command -v hipcc)" \
  -D CMAKE_PREFIX_PATH="$torch_root/share/cmake;$ROCM_PATH" \
  -D CMAKE_INSTALL_PREFIX="$LAMMPS_INSTALL" \
  -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=OFF \
  -D PKG_USER-MFFTORCH=ON -D MFF_USE_HIP=ON -D MFF_ENABLE_VIRIAL=OFF \
  -D PKG_KOKKOS=ON -D Kokkos_ENABLE_HIP=ON -D Kokkos_ENABLE_SERIAL=ON \
  -D Kokkos_ENABLE_OPENMP=OFF -D Kokkos_ARCH_AMD_GFX926=ON \
  -D AMDGPU_TARGETS=gfx926 -D GPU_TARGETS=gfx926 \
  -D CMAKE_HIP_ARCHITECTURES=gfx926
cmake --build build-lammps-ictc --parallel "${SLURM_CPUS_PER_TASK:-4}"
cmake --install build-lammps-ictc
```

```bash
sbatch --account=YOUR_ACCOUNT --cpus-per-task=8 \
  "$ICTC_REPO/examples/k100_job.sbatch" bash build_lammps.sh
```

Your site's host compiler, MKL headers and Torch wheel link paths may need
additional CMake arguments. If imported Torch targets reference absent
build-machine paths, follow the task-local adjustment in the build guide;
the package option alone cannot repair an incompatible vendor environment.

### Expose the matching Python bindings and shared library

After installation, put this in `lammps_env.sh` and source it **inside each
LAMMPS job**, after loading the build's MPI environment:

```bash
torch_root="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')"
export PYTHONPATH="$LAMMPS_SOURCE/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$LAMMPS_INSTALL/lib:$OPENMPI_HOME/lib:$torch_root/lib:$(dirname "$torch_root")/torch.libs:${LD_LIBRARY_PATH:-}"
```

`OPENMPI_HOME` is the example site's MPI prefix; replace it for your MPI.
Use `lib64` if that is your install layout. Both the `lammps` Python package
and its matching shared library are needed, as described in the
[LAMMPS Python installation documentation](https://docs.lammps.org/Python_install.html).
Do not substitute an unrelated pip LAMMPS wheel for this custom build.

## 10. Export and run in LAMMPS

### Export the deployment bundle

For a trained short-range or reciprocal ICTC checkpoint:

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/k100_job.sbatch" \
  mff-export-core --checkpoint models/hco-short.pth \
    --elements H C O --out models/core.pt
```

Use `models/hco-long.pth` to export the learned reciprocal model. Its
electrostatic settings come from the checkpoint; the same pair style
evaluates the complete trained energy and forces. No separate LAMMPS
`kspace_style` is required for this model's built-in reciprocal correction.

For MACE-MH-1 on copper:

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/k100_job.sbatch" \
  mff-export-core --checkpoint models/mace-mh-1-ictc-hp.pt \
    --head omat_pbe --elements Cu --out models/mh1-core.pt
```

The export default is `--backend native --inference-mode performance`,
using FP32 and including E0. For larger systems choose
`--inference-mode capacity`, for example with `--out models/core-capacity.pt`.
It bounds intermediate storage and recomputes selected activations; it
retains the model weights, neighbors and FP32 arithmetic. It can be slower.

Keep all four output files together:

```text
core.pt
core.pt.checkpoint.pth
core.pt.json
core.pt.meta
```

This is an archive plus runtime metadata and weights, not a renamed Python
checkpoint. It requires the installed MACE-ICTC runtime. Inspect `core.pt.json`
for the trained cutoff, element order, selected head and inference settings.

### Prepare a LAMMPS input

The supplied [`in.hco`](../examples/in.hco) expects `system.data` with type
1=H, type 2=C and type 3=O, and a checkpoint with cutoff **4.5 Å**:

```bash
cp "$ICTC_REPO/examples/in.hco" in.mfftorch
```

Its essential lines are:

```lammps
units metal
atom_style atomic
atom_modify map array
boundary p p p
read_data system.data
neighbor 0.5 bin
pair_style mff/torch/kk 4.5 cuda
pair_coeff * * models/core.pt H C O
thermo_style custom step atoms pe fmax
run 0
```

`pair_coeff` symbols map **LAMMPS atom types in order**, not atomic numbers.
Use the matching `--elements` order during export. Set the pair-style cutoff
to the checkpoint value; for the MH-1 copper example, use its recorded
cutoff and `pair_coeff * * models/mh1-core.pt Cu` with one Cu atom type.

### One GPU and eight GPUs

Create `run_lammps.sh` in the working directory:

```bash
#!/usr/bin/env bash
set -euo pipefail
source ./lammps_env.sh
exec mpirun --bind-to none -np "${SLURM_NTASKS:-1}" mff-lammps -in in.mfftorch
```

```bash
# One GPU.
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/k100_job.sbatch" \
  bash run_lammps.sh

# Eight GPUs on one node: one MPI rank per GPU.
sbatch --account=YOUR_ACCOUNT --ntasks=8 --gres=dcu:8 --mem=256G \
  "$ICTC_REPO/examples/k100_job.sbatch" bash run_lammps.sh
```

The public launcher enables Kokkos, full neighbors and Newton off. It binds
the HP runtime to Kokkos's selected device and performs the feature/adjoint
exchanges needed by each interaction. The qualified MPI workflow is one
node, one rank per GPU, a single communicator and nonempty owned domains.
Use enough atoms per rank; eight ranks are not appropriate for a tiny molecule.

For the plain interface, change the input to `pair_style mff/torch ...` and
launch `mff-lammps --plain -in in.mfftorch`. Kokkos is the performance default.

### Starting dynamics

`run 0` evaluates energy and forces and the example writes `forces.dump`.
After checking those values, a short NVE example can replace `run 0` with:

```lammps
velocity all create 300.0 4928459 mom yes rot no dist gaussian
timestep 0.00025
fix integrate all nve
run 100
```

For `metal` units this timestep is 0.25 fs; choose temperature and timestep
for the potential and system you actually validated. Native Kokkos virial
accumulation defaults off. Pressure, NPT and stress require separate
qualification; successful energy/force evaluation does not validate them.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Large energy offset with reasonable forces | Pass the correct E0 convention; avoid omitting or adding E0 twice. |
| Prepared H5 cutoff differs from training | Reprocess at the actual model cutoff. |
| `generated` rejects a model | Check FP32, native full CG, frozen TP base weights and supported architecture; use an explicit reference route for diagnosis. |
| Converted checkpoint will not load | Check payload type, architecture support, first interaction, readout, radial widths and CG basis. |
| Long-range configuration fails | Check periodic cells, rank ≤ backbone lmax, mesh/screening options and absence of node padding. |
| `mff/torch/kk` is unknown | Register/build USER-MFFTORCH and KOKKOS, then expose that same shared LAMMPS library. |
| MPI import errors, hangs or serial-library errors | Match MPI across LAMMPS, mpi4py and the launcher; inspect library search ordering. |
| First evaluation is much slower | Native HIP JIT and initial model/graph setup are cold-start work. |
| GPU OOM | Reduce training microbatch size; use capacity export for inference; include ghost, Kokkos and RCCL memory in the budget. |

For details, see the [HP API](HP_API.md), [electrostatics conventions](ELECTROSTATICS.md),
[LAMMPS build guide](../lammps_user_mfftorch/docs/BUILD_AND_RUN.md), and
[upstream Kokkos build options](https://docs.lammps.org/Build_extras.html#kokkos-package).

## Dispersion and FMM

This K100 distribution supports short-range MACE and learned mesh/FFT
electrostatics. Dispersion (pairwise C6/MBD) and tree FMM implementations are
excluded. Existing checkpoints with dispersion disabled retain their architecture
metadata and load normally. Checkpoints requiring those removed contributions
are rejected during reconstruction or deployment loading.
