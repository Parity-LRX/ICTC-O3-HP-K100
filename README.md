# ICTC-O3-HP-K100

Cartesian tensor operators for K100 (`gfx926`), with HIP kernels for atomistic
energy/force training and inference. Version **0.1.0**.

Distribution: `ictc-o3-hp-k100`. Python packages: `ictc_o3` for standalone
operators and `mace_ictc` for models, training and conversion.

## Python API

The two packages live side by side and are installed together:

```text
ictc_o3/          # Public Cartesian tensor operators
mace_ictc/        # MACE-compatible models, training and interfaces
```

```python
from ictc_o3 import (
    EdgeTensorProduct, GraphMessage, PackedGraphMessage,
    direction_harmonics, create_symmetric_contraction,
    configure_symmetric_contraction, channel_linear, normalized_silu,
)
from mace_ictc.hp import create_model, configure_model, energy_and_forces
from mace_ictc.training import ForceTrainer
```

See the [operator API](docs/O3_API.md) for shapes, gradients and standalone usage,
and the [model API](docs/HP_API.md) for training, inference and conversion.
Existing operator imports from `mace_ictc.hp` remain compatible.

## Installation

Use an isolated environment with a compatible K100 vendor PyTorch, HIP/DTK
and Triton stack, then install:

```bash
git clone https://github.com/Parity-LRX/ICTC-O3-HP-K100.git
cd ICTC-O3-HP-K100
python -m pip install -e '.[hp,train,conversion]'
```

The `hp` extra supplies Ninja and preserves the installed vendor Triton.
GPU execution and just-in-time (JIT) compilation need the matching HIP compiler and runtime.
On Slurm clusters, run computation through `sbatch` on an allocated GPU.

## Features

- Tensor products, graph aggregation, direction harmonics and symmetric contractions.
- HIP kernels for first-order backward and mixed second derivatives in energy/force training.
- Streamed inference gates, chunked backward kernels and grouped element-conditioned linear maps.
- Nonlinear MACE-MH-1 conversion with all original readout heads retained.
- LAMMPS interfaces and model export utilities.

```python
from mace_ictc.hp import create_model, configure_model

model = create_model(channels=64, lmax=2, num_interactions=2, correlation=3)
configure_model(
    model, training=True, policy="generated", layout="fused_pack",
)
```

Generated eager training uses `training_message_linear_layout="node_component"`
by default in both the HP API and `ForceTrainer`/CLI. Set it to `"einsum"` (CLI:
`--training-message-linear-layout einsum`) to select the previous layout.

See the [API guide](docs/HP_API.md) for training, inference and conversion.
The current MH-1 foundation adapter supports frozen-weight force inference;
full MH-1 fine-tuning requires a training adapter.

## Multi-GPU results

Measured on one node with 64 GiB K100 GPUs, FP32 arithmetic and TensorFloat-32
(TF32) disabled. Molecular dynamics (MD) timing includes integration, neighbor
rebuilding/sorting, energy, forces, communication and release of unused PyTorch
allocator cache. The neighbor list is rebuilt every step in these workloads.
Scaling uses one MPI rank and two CPU threads per GPU; 3 warmup steps, then
three consecutive 20-step windows. Each step uses synchronized wall time and
the maximum across MPI ranks. Reported scaling time is the median of the three
window means; startup and JIT are excluded.

The measured interface uses LAMMPS
[ML-IAP](https://docs.lammps.org/pair_mliap.html) for machine-learning interatomic
potentials, with FP32 GPU feature exchange through RCCL collectives. These
results describe that eager interface; standalone `.pt2` deployment has its
own backend requirements in the [LAMMPS guide](lammps_user_mfftorch/docs/BUILD_AND_RUN.md).

### C64 scaling

Remeasured with 0.1.0 on 2026-10-21: C64, hidden/edge lmax=2,
2 interactions, full Clebsch–Gordan (CG) correlation=3, randomly initialized
weights for H/C/O, cutoff 4.5 Å and about 50 neighbors/atom. Grouped
element-conditioned linear maps; forward/backward edge chunks
1,048,576/32,768. C64 means 64 channels per irrep degree; `lmax` is the maximum
angular degree. Correlation and interaction depth are separate model settings;
see the [API terminology](docs/HP_API.md#terminology-and-notation).

Strong scaling, fixed **262,144 atoms**:

| GPUs | ms/step | Speedup | Efficiency |
| ---: | ---: | ---: | ---: |
| 1 | 2,128.899 | 1.000× | 100.00% |
| 2 | 1,125.568 | 1.891× | 94.57% |
| 4 | 566.008 | 3.761× | 94.03% |
| 8 | 301.215 | **7.068×** | **88.35%** |

Weak scaling, fixed **131,072 atoms/GPU** and unchanged domain size per GPU:

| GPUs | Total atoms | ms/step | Throughput speedup | Efficiency |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 131,072 | 1,055.564 | 1.000× | 100.00% |
| 2 | 262,144 | 1,114.135 | 1.895× | 94.74% |
| 4 | 524,288 | 1,132.590 | 3.728× | 93.20% |
| 8 | 1,048,576 | 1,172.413 | **7.203×** | **90.03%** |

### MACE-MH-1 scaling

Measured with 0.1.0 on 2026-10-21: pretrained `omat_pbe` head, with all six
original heads retained; periodic face-centered cubic (FCC) Cu, lattice
constant 3.615 Å, cutoff 6 Å and 78 neighbors/atom. Projected-feature FP32 RCCL
exchange; forward, backward and radial edge chunks 262,144/32,768/32,768.
Same timing protocol as above.
Every GPU count passed 256-atom original-MACE energy/force checks and
per-rank ghost-feature and reverse-gradient communication checks.

Strong scaling, fixed **65,536 atoms**:

| GPUs | ms/step | Speedup | Efficiency |
| ---: | ---: | ---: | ---: |
| 1 | 6,907.408 | 1.000× | 100.00% |
| 2 | 3,505.261 | 1.971× | 98.53% |
| 4 | 1,774.945 | 3.892× | 97.29% |
| 8 | 913.766 | **7.559×** | **94.49%** |

Weak scaling, fixed **32,768 atoms/GPU** and unchanged domain size per GPU:

| GPUs | Total atoms | ms/step | Throughput speedup | Efficiency |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32,768 | 3,437.113 | 1.000× | 100.00% |
| 2 | 65,536 | 3,506.372 | 1.960× | 98.02% |
| 4 | 131,072 | 3,524.422 | 3.901× | 97.52% |
| 8 | 262,144 | 3,543.252 | **7.760×** | **97.00%** |

For both models, strong efficiency is `T₁(N)/(p×Tₚ(N))`; weak efficiency is
`T₁(n)/Tₚ(p×n)`, where `p` is the GPU count, `N` the global atom count, `n` the
atoms per GPU and `Tₚ` the measured step time. The three consecutive windows
do not provide a confidence interval from independent runs.

### Eight-GPU capacity

ML-IAP with FP32 RCCL feature exchange;
50 timed microcanonical (NVE) steps after warmup, 0.25 fs timestep. Atoms are global physical
atoms, excluding ghost copies; times are per-step medians.

| Model | Atoms | Mean neighbors | s/step |
| --- | ---: | ---: | ---: |
| C64, lmax=2, 2 interactions, correlation=3 | **3,241,792** | 49.73 | 3.694 |
| MACE-MH-1, `omat_pbe` head | **1,146,880** | 78 | 16.083 |

The C64 capacity case uses randomly initialized H/C/O weights and a 4.5 Å cutoff;
MH-1 uses pretrained weights, periodic FCC Cu and a 6 Å cutoff. These are
short capacity checks near the memory limit; capacity depends on the model,
density and neighbor count, and does not establish long-trajectory stability.

## License

Derived from [ICTC-O3-HP](https://github.com/Parity-LRX/ICTC-O3-HP).
Existing PolyForm Noncommercial 1.0.0 and MIT license scopes are retained;
see [LICENSE_SCOPE.md](LICENSE_SCOPE.md) and [NOTICE](NOTICE).
Pretrained model weights have their own upstream licenses.
