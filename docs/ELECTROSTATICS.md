# Native K100 reciprocal electrostatics

The K100 mesh backend accelerates the original ICTC reciprocal functional
for monopoles, Cartesian dipoles and Cartesian quadrupoles. It supports
arbitrary atom counts and positive source-channel counts, CIC/TSC/PCS
assignment, periodic or padded-slab boundaries, batched triclinic cells,
and FP32/FP64 on Hygon K100 (`gfx926`). Mesh size is a positive integer;
the current ICTC interface uses a cubic mesh.

## Fixed physical screening

New `MeshElectrostatics` instances and the `create_model(electrostatics=...)`
factory use `smearing_sigma=1.0`, measured in the same length unit as positions
and cells. Gaussian screening multiplies the reciprocal kernel by
`exp(-sigma**2 * k**2 / 2)`, equivalently `alpha = 1/(sqrt(2)*sigma)`.
The length stays fixed when the box is replicated or strained. It is a model
hyperparameter; changing it after training changes the potential.

Set `smearing_sigma=None` only to reproduce legacy box-dependent screening
(`alpha = 10/L_min`). Existing checkpoint metadata without the new field
retains that legacy rule. Lower-level model constructors and the training
CLI also preserve their old default; pass `long_range_smearing_sigma=1.0`
or `--long-range-smearing-sigma 1.0` to opt in. Fixed screening requires
`mesh_fft` and Gaussian screening enabled (`mesh_fft_full_ewald=True`).
For a bare kernel, use `full_ewald=False, smearing_sigma=None`.

When replicating a periodic cell, increase mesh dimensions by the same factor
to retain physical grid spacing. A fixed mesh size is not a fixed spatial
resolution. The cubic-mesh API uses the largest cell dimension to bound
spacing; an elongated box can therefore need a larger mesh. Converge mesh
resolution for the chosen source ranks and smoothing length. In particular,
high multipole derivatives can amplify mesh aliasing even when energy errors
look small.

Fixed-sigma models retain their reciprocal energy inside exported graphs.
The separate C++ reciprocal solver currently uses legacy box-dependent
screening, so fixed-sigma models do not delegate sources to that solver.

## Assignment implementation benchmark

![Single-K100 reciprocal energy-and-force throughput and peak allocated memory, comparing original ICTC Torch operations with native HIP assignment and real FFTs. Error bars show the range of three timing-round medians.](assets/electrostatics-k100.svg)

For 1,048,576 atoms with monopoles, dipoles and quadrupoles on a 128³ mesh,
energy-and-force evaluation takes 694 → 338 ms. Peak PyTorch allocation
falls from 6.43 to 1.22 GiB. At 262,144 atoms on a 96³ mesh, the corresponding
monopole evaluation takes approximately 38 → 8 ms. At the same size, a
monopole energy-plus-force loss backward and AdamW source update takes
75 → 27 ms; with dipoles and quadrupoles it takes 399 → 266 ms.

These measurements isolate the reciprocal module with one prescribed
source channel, eager FP32 and TF32 disabled. Source prediction, the MACE
backbone, neighbor construction and multi-GPU communication are excluded.
This compatibility benchmark uses legacy box-dependent screening
(`smearing_sigma=None`), rather than the new fixed-length factory default.
The curves connect measured sizes; endpoints do not establish capacity
limits. Error bars show timing variation, rather than confidence intervals.
Smaller workloads show more host-timing variation.

## Validation

Fixed-length screening is checked separately from implementation parity:

- Two physical lengths and all three source ranks agree with the full-FFT
  reference through mixed second derivatives. Mesh refinement is compared
  with an independent continuous structure-factor calculation.
- Eightfold periodic replication preserves energy per cell and per-atom
  forces at identical physical grid spacing. Three FP32 whole models with
  two source channels also preserve the cell virial, with relative errors
  below `1.1e-6`; force differences are below `3.4e-6`.
- Nine strain components pass independent FP64 finite differences, including
  cubic cells with equal lattice-vector lengths.
- Far-field forces for q–q, q–mu, q–Q, mu–mu, mu–Q and Q–Q recover their
  expected distance powers and signs. At sigma=1, a 96-unit periodic box
  and a 192³ PCS mesh, the largest discrepancy from the isolated asymptote
  over separations 6, 9 and 12 is below 0.86%. This includes periodic-image
  effects and is not a universal error guarantee.
- Fixed-length and legacy checkpoints reload their respective screening
  rules. Whole-model energy/force/virial-loss parameter gradients match the
  Torch reference for monopole, dipole and quadrupole heads.

K100 validation covers 12 assignment cases, 14 electrostatics configurations,
24 full-spectrum FFT comparisons, three whole-model energy/force/stress-loss
gradient comparisons, and scalar/multipole checkpoint round trips with
nondefault source widths and correlation. Tests include FP32/FP64,
CIC/TSC/PCS, odd meshes, triclinic and padded-slab cells, empty atoms and
empty graph members. Independent finite differences check first and mixed
second derivatives. Channel-first adjoints, graph-capture fallback and
deterministic-execution guards are checked separately.

FP32 comparisons against an FP64 oracle use relative L2 error at most
`2e-5` and maximum absolute error at most
`2e-5 * max(1, max(abs(reference)))`. FP64 comparisons use elementwise
absolute/relative tolerance `2e-8`. Near-cancelling FP32 cell-gradient
components also vary between repeats of the original atomic-reduction
implementation. The million-atom force comparison has relative L2 error
approximately `1.7e-6` for both the original and native FP32 implementations.

## Standalone operator

```python
import torch
from ictc_o3 import MeshElectrostatics

solver = MeshElectrostatics(
    mesh_size=64, assignment="pcs", boundary="periodic",
    neutralize=True, full_ewald=True, smearing_sigma=1.0, backend="hip",
).cuda()

# positions: [N,3]; batch: int64 [N]; cell: [B,3,3], lattice vectors in rows.
# q: [N,S]; mu: [N,S,3]; Q: [N,S,3,3]. Omit mu/Q for monopoles.
positions = positions.requires_grad_(True)
atom_energy = solver(positions, batch, cell, q, dipole=mu, quadrupole=Q)
energy = atom_energy.sum()
forces = -torch.autograd.grad(energy, positions, create_graph=True)[0]
loss = (energy - target_energy).square() + (forces - target_forces).square().mean()
loss.backward()
```

Inputs share a floating dtype and device. Use `solver.double()` with FP64
inputs. Scalar sources are neutralized per graph; dipoles and quadrupoles
are not neutralized. Empty inputs and empty graph-batch members are
supported. Scalar energies use the requested `energy_partition`; multipole
energies are partitioned uniformly over each graph's atoms.

Positions, moments, learned Green-filter parameters and cells remain
differentiable. Cell derivatives support strain-based stress evaluation
with the application's sign and volume convention. Native backward and
mixed second derivatives cover energy-plus-force training. For third and
higher derivatives, select `backend="reference"`.

Low-level assignment and its adjoint are also public:

```python
from ictc_o3 import mesh_spread, mesh_gather

mesh = mesh_spread(
    fractional_positions, batch, source,
    mesh_size=64, num_graphs=num_graphs, assignment="pcs", backend="hip",
)
sampled = mesh_gather(
    fractional_positions, batch, mesh,
    mesh_size=64, num_graphs=num_graphs, assignment="pcs", backend="hip",
)
```

The default mesh layout is `[B,M,M,M,C]`. `channels_first=True` uses
`[B,C,M,M,M]`. Fractional coordinates follow the original boundary
preparation: `slab` wraps x/y and clamps z mesh indices after vacuum padding.

## Learned MACE-ICTC model

```python
from mace_ictc.hp import create_model, configure_model

model = create_model(
    channels=64, lmax=2, num_interactions=2, correlation=3,
    electrostatics={
        "mesh_size": 64, "source_channels": 2,
        "max_multipole_l": 2,  # 0: monopoles; 1: add dipoles; 2: add quadrupoles
        "assignment": "pcs", "smearing_sigma": 1.0,
    },
)
configure_model(
    model, training=True, policy="generated", layout="fused_pack",
    electrostatics_backend="hip",
)
```

`electrostatics=None` preserves the short-range factory behavior. A mapping
enables `reciprocal-spectral-v1` with a periodic mesh, Gaussian screening
at a fixed length of 1.0 and reciprocal-only energy by default. It accepts the original long-range
factory options without the `long_range_` prefix. `max_multipole_l` must not
exceed the backbone `lmax` or 2. Equivariant readouts and channel mixers
remain trainable. The scalar energy gate starts at zero; the multipole
source gate starts at 0.1.

Existing modules and checkpoints can select execution explicitly:

```python
from mace_ictc.hp import configure_electrostatics

configure_electrostatics(model, backend="hip", fuse_multipole_fft=True)
configure_electrostatics(model, backend="reference")
```

`auto` selects native assignment on K100 FP32/FP64 and the Torch reference
elsewhere. `hip` rejects unsupported devices/dtypes when evaluated.
Configuration does not freeze parameters, cast tensors or compile kernels.
`configure_model(policy="reference")` restores reference electrostatics;
export preparation selects reference mesh operations for graph capture.
The HIP extension is compiled lazily during eager execution.

The training CLI also accepts `--long-range-mesh-fft-reciprocal-only`.
Together with `--long-range-mesh-fft-full-ewald`, it selects the screened
reciprocal functional. This flag is saved and restored in checkpoint
metadata. A missing flag retains the existing K100 checkpoint default;
runtime backend choices add no state-dict parameters.
`long_range_smearing_sigma` is saved and restored alongside these options.
`ForceTrainer(..., config=cfg, electrostatics_backend="hip")` and CLI
`--electrostatics-backend hip` explicitly require the eager K100 backend.
Pass the matching `ModelConfig` to the trainer when writing checkpoints.

## Functional and implementation

With hats denoting assigned-grid Fourier transforms:

```text
S(k) = q_hat(k) + i k · mu_hat(k) - 1/2 k · Q_hat(k) · k
E    = 1/(2V) sum_{k != 0, channels} G(k) |S(k)|² / |W(k)|²
```

`G(k)` is the original Poisson or learned Poisson kernel with configured
Gaussian screening. `W(k)` is the matching B-spline assignment window.
FFT normalization, quadrupole sign/factor, neutralization and graph
partitioning follow the original ICTC implementation. No additional
Coulomb unit-conversion constant is inserted; sources and prefactors use
the caller's unit system.

The forward Fourier transform uses `exp(-i*k.r)`, and the API preserves the
original `+i*k.mu` source convention. If standalone sources represent a
physical point-dipole density `-p.grad(delta)`, pass `dipole=-p`. The quadrupole
term uses the Cartesian second-moment convention shown above, with its factor
of one half. Learned source heads fit these conventions during training.

HIP kernels calculate weights and derivatives while loading atoms,
avoiding `[N,stencil,C]` source/gradient temporaries. Their adjoints include
coordinate Hessian-vector products and source/mesh cross terms. The
multipole path can spread all components once and use one real-FFT batch.
Both folded mirror wavevectors are retained at Nyquist boundaries, so
triclinic cells and unscreened kernels preserve the full-spectrum functional.
FFTs use PyTorch's vendor HIP FFT backend. Live inputs, topology, moments,
cell-dependent spectral weights and learned parameters are not cached.
Floating atomic reductions are not bitwise deterministic.

This API implements a screened reciprocal correction. Full fixed-charge
Ewald additionally requires real-space, self and background terms. The
existing scalar exact-Ewald option retains its reference real-space path;
this change does not accelerate that dense pairwise calculation. Padded
slab boundaries retain the original approximation, rather than an exact
two-dimensional Ewald sum. Mesh resolution and padding need physical
convergence checks. Only the legacy `smearing_sigma=None` estimate uses the
shortest periodic lattice vector and has a nondifferentiable minimum at tied
shortest lengths. Fixed-sigma screening removes that minimum from the
reciprocal energy.

Each graph must supply its physical atoms once. This backend uses one
device per graph and does not implement distributed FFTs or halo-charge
exchange. Independent-graph DDP can use a local mesh per rank.
Domain-decomposed MD needs a global long-range solver. Existing C++ source
export/deployment paths are separate from these eager HIP kernels.
