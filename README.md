# ICTC-O3-HP-K100

High-performance **ICTC-O3 operators optimized for Hygon K100 GPUs**, together
with the **MACE-ICTC framework** for atomistic modeling, training and inference.

Version **0.1.0**. Distribution: `ictc-o3-hp-k100`.
Python packages: `ictc_o3` for standalone operators and `mace_ictc` for the
MACE-ICTC framework.

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

Measured on a single node with 64 GiB Hygon K100 GPUs, FP32 arithmetic and
TensorFloat-32 (TF32) disabled. The figures show complete MD step throughput,
including integration, neighbor rebuilding, energy, forces and communication.
Throughput is `global physical atoms / seconds per step`, reported as
atom-steps/s; ghost copies are excluded from the atom count. `k` means thousand.

These measurements use the eager LAMMPS
[ML-IAP](https://docs.lammps.org/pair_mliap.html) interface with FP32 RCCL feature
exchange. Standalone `.pt2` deployment has its own backend requirements in the
[LAMMPS guide](lammps_user_mfftorch/docs/BUILD_AND_RUN.md).

### C64 scaling

![C64 on 1, 2, 4 and 8 K100 GPUs: strong scaling reaches 7.07 times speedup; weak scaling reaches 894 thousand atom-steps per second with 90.03 percent efficiency.](docs/assets/c64-scaling.svg)

C64, hidden/edge `lmax=2`, 2 interactions, full Clebsch–Gordan (CG)
`correlation=3`; randomly initialized H/C/O weights, cutoff 4.5 Å and about
50 neighbors/atom. Grouped element-conditioned linear maps and forward/backward
edge chunks of 1,048,576/32,768. Strong scaling fixes 262,144 atoms; weak
scaling uses 131,072 atoms/GPU with unchanged domain size per GPU.

<details>
<summary>Exact step times, throughput and scaling data</summary>

Strong scaling: fixed total atom count; speedup is the one-GPU time divided by the multi-GPU time.

| GPUs | Global atoms | ms/step | Throughput (atom-steps/s) | Speedup | Efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 262,144 | 2,128.899 | 123,136 | 1.000× | 100.00% |
| 2 | 262,144 | 1,125.568 | 232,899 | 1.891× | 94.57% |
| 4 | 262,144 | 566.008 | 463,146 | 3.761× | 94.03% |
| 8 | 262,144 | 301.215 | 870,288 | 7.068× | 88.35% |

Weak scaling: fixed atoms per GPU; speedup is total throughput divided by the one-GPU throughput.

| GPUs | Global atoms | ms/step | Throughput (atom-steps/s) | Speedup | Efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 131,072 | 1,055.564 | 124,173 | 1.000× | 100.00% |
| 2 | 262,144 | 1,114.135 | 235,289 | 1.895× | 94.74% |
| 4 | 524,288 | 1,132.590 | 462,911 | 3.728× | 93.20% |
| 8 | 1,048,576 | 1,172.413 | 894,374 | 7.203× | 90.03% |

</details>

### MACE-MH-1 scaling

![Pretrained MACE-MH-1 on 1, 2, 4 and 8 K100 GPUs: strong scaling reaches 7.56 times speedup; weak scaling reaches 74.0 thousand atom-steps per second with 97.00 percent efficiency.](docs/assets/mh1-scaling.svg)

Pretrained `omat_pbe` head, with all six original heads retained. Periodic
face-centered cubic (FCC) Cu, lattice constant 3.615 Å, cutoff 6 Å and
78 neighbors/atom. Projected-feature FP32 RCCL exchange; forward, backward and
radial edge chunks 262,144/32,768/32,768. Strong scaling fixes 65,536 atoms;
weak scaling uses 32,768 atoms/GPU with unchanged domain size per GPU.
Every GPU count passed 256-atom original-MACE energy/force checks and
per-rank ghost-feature and reverse-gradient communication checks.

<details>
<summary>Exact step times, throughput and scaling data</summary>

Strong scaling: fixed total atom count; speedup is the one-GPU time divided by the multi-GPU time.

| GPUs | Global atoms | ms/step | Throughput (atom-steps/s) | Speedup | Efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 65,536 | 6,907.408 | 9,488 | 1.000× | 100.00% |
| 2 | 65,536 | 3,505.261 | 18,696 | 1.971× | 98.53% |
| 4 | 65,536 | 1,774.945 | 36,923 | 3.892× | 97.29% |
| 8 | 65,536 | 913.766 | 71,721 | 7.559× | 94.49% |

Weak scaling: fixed atoms per GPU; speedup is total throughput divided by the one-GPU throughput.

| GPUs | Global atoms | ms/step | Throughput (atom-steps/s) | Speedup | Efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32,768 | 3,437.113 | 9,534 | 1.000× | 100.00% |
| 2 | 65,536 | 3,506.372 | 18,691 | 1.960× | 98.02% |
| 4 | 131,072 | 3,524.422 | 37,190 | 3.901× | 97.52% |
| 8 | 262,144 | 3,543.252 | 73,984 | 7.760× | 97.00% |

</details>

#### MD timing protocol

Scaling was measured with version 0.1.0 on 2026-10-21: one MPI rank and two CPU
threads per GPU, 3 warmup steps and three consecutive 20-step windows. Each
step uses synchronized wall time and the maximum across MPI ranks; the reported
time is the median of the three window means. Startup and JIT compilation are
excluded. Neighbor rebuilding/sorting and release of unused PyTorch allocator
cache are included; the neighbor list is rebuilt every step in these workloads.
The three consecutive windows do not provide a confidence interval from
independent runs.

Strong efficiency is `T₁(N)/(p×Tₚ(N))`; weak efficiency is `T₁(n)/Tₚ(p×n)`, where
`p` is the GPU count, `N` the global atom count, `n` the atoms per GPU and `Tₚ`
the measured step time. C64 means 64 channels per irrep degree; correlation and
interaction depth are separate settings. See the
[API terminology](docs/HP_API.md#terminology-and-notation).

### Eight-GPU capacity

![Demonstrated eight-K100 capacity: C64 runs 3,241,792 atoms at 3.694 seconds per step, and pretrained MACE-MH-1 runs 1,146,880 atoms at 16.083 seconds per step.](docs/assets/eight-gpu-capacity.svg)

ML-IAP with FP32 RCCL feature exchange; 50 timed microcanonical (NVE) steps
after warmup, 0.25 fs timestep. Times are per-step medians; capacity throughput
is derived from the reported, rounded step times. These are short runs near
the memory limit. Capacity depends on the model, density and neighbor count,
and does not establish long-trajectory stability. The two cases use the model
and material configurations shown above.

<details>
<summary>Capacity measurements</summary>

| Model | Global atoms | Mean neighbors | s/step | Throughput (atom-steps/s) |
| --- | ---: | ---: | ---: | ---: |
| C64 | 3,241,792 | 49.73 | 3.694 | ~877,583 |
| MACE-MH-1 | 1,146,880 | 78 | 16.083 | ~71,310 |

</details>

## Energy-and-force training

![Four selected single-K100 HP training workloads: the default node_component message-linear layout raises absolute training throughput by up to 1.70 times and reduces step time by up to 41.2 percent.](docs/assets/training-throughput.svg)

The comparison isolates the HP message-linear layout change from `einsum` to
the current default `node_component`, with the same model, parameters and
FP32 arithmetic. Each step includes forward evaluation, forces with
`create_graph=True`, the complete energy-plus-force loss backward including
second derivatives, and an AdamW update.

The four plotted cases are selected large-graph workloads from a 44-point
screen. The largest measured gain is 1.70× training throughput, equivalent
to 41.2% less time per step. No point in that screen showed a step-time
regression over 5%; the median throughput gain across all 44 points was 3.2%.
The layout passed 18 operator checks and 9 whole-model gradient comparisons
against the independent HP reference.

Training measurements use one K100, two interactions, `correlation=3`, periodic
H/C/O with spacing 2 Å and cutoff 4.5 Å, and a fixed neighbor graph. There are
10 warmup steps and three 5-step rounds; the reported time is the median of
the round medians. Data loading, neighbor rebuilding, compilation, profiling
and multi-GPU communication are outside this training timing. These training
throughputs have a different workload and timing scope from the MD figures.

<details>
<summary>Training step times and throughput</summary>

| Channels / lmax / atoms | Previous HP, ms/step | Default HP, ms/step | Previous → default throughput (atom-steps/s) | Throughput gain |
| --- | ---: | ---: | ---: | ---: |
| 256 / 1 / 32,768 | 1,877.40 | 1,103.57 | 17,454 → 29,693 | 1.701× |
| 128 / 1 / 65,536 | 1,861.39 | 1,386.72 | 35,208 → 47,260 | 1.342× |
| 128 / 3 / 16,384 | 2,239.03 | 1,838.09 | 7,317 → 8,914 | 1.218× |
| 256 / 3 / 8,192 | 1,985.31 | 1,692.95 | 4,126 → 4,839 | 1.173× |

</details>

## License

Derived from [ICTC-O3-HP](https://github.com/Parity-LRX/ICTC-O3-HP).
Existing PolyForm Noncommercial 1.0.0 and MIT license scopes are retained;
see [LICENSE_SCOPE.md](LICENSE_SCOPE.md) and [NOTICE](NOTICE).
Pretrained model weights have their own upstream licenses.
