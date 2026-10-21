# ICTC-O3-HP-K100 API

Native monopole, dipole and quadrupole mesh operations and model integration
are described in the [electrostatics guide](ELECTROSTATICS.md).

`ictc_o3` provides standalone operator APIs, and `mace_ictc.hp` provides
whole-model APIs. These sibling Python packages are installed together with
`ictc-o3-hp-k100`. See the [O3 operator overview](O3_API.md) for a quick reference.
Existing operator imports from `mace_ictc.hp` remain compatible. Users do not
need to call internal functions such as `_source`, `_plan` or GPU kernel launchers
in `models/`.

## Terminology and notation

| Term | Meaning in this API |
| --- | --- |
| O(3), SO(3) | O(3) includes rotations and reflections; SO(3) includes proper rotations. An O(3) irreducible representation (irrep) has degree `l` and parity `p`. |
| Degree and parity | An irrep of degree `l` has `2l+1` angular components. Natural parity means `p=(-1)^l`. The standalone TP factories use this family, rather than accepting arbitrary `(l,p)` irreps. |
| `L`, `C`, `P` | Maximum irrep degree, channel count (irrep multiplicity), and number of coupling paths. `input_lmax`, `edge_lmax` and `output_lmax` can differ. |
| `N`, `E`, `B` | Node count, directed edge count and number of graphs in a batch. In distributed buffers, `N` includes owned and ghost rows unless stated otherwise. Global physical atom counts exclude ghost copies. |
| Angular basis and layout | ICTC uses Cartesian harmonic blocks. Packing changes memory order, not the angular basis. In `[*,C,m]`, `m=2l+1` denotes the block width, not a magnetic quantum number. |
| TP, CG | Tensor product and Clebsch–Gordan coupling. A TP path is `(l_in,l_edge,l_out)`. The edge TP couples each feature channel to a single-channel angular input; it retains paths for later channel mixing. |
| SC, correlation, U | Symmetric contraction, its maximum polynomial order in aggregated features, and its fixed higher-order coupling tensors. Correlation is not interaction depth or a statistical correlation coefficient. |
| Full CG / reduced CG | Names for this implementation's coupling-basis construction choices. The public factories use the existing full CG bridge-U construction (`use_reduced_cg=False`). SC retains its internal e3nn basis bridge. |
| Gates | Edge-dependent radial-network outputs multiplying TP paths. Frozen radial-network parameters still produce gates that depend on geometry. Coordinate derivatives must pass through those gates. |
| MLP, JIT, FX | Multilayer perceptron; just-in-time compilation; PyTorch's graph representation and tracing tools. The radial MLP maps edge descriptors to gates. |
| VJP, mixed second derivative | A vector–Jacobian product is a reverse-mode derivative. Force-loss training requires mixed derivatives of energy with respect to coordinates and model parameters, without materializing the full Hessian. Option names containing `adjoint` refer to these backward computations. |
| CSR, GEMM | Compressed sparse row grouping of edges by destination node; general matrix–matrix multiplication. Neither changes the model's mathematical definition. |
| E0 | Per-element atomic-energy offsets from the checkpoint or training fit. These offsets must be included consistently when comparing total energies. |
| FP32, FP64, TF32 | 32-bit and 64-bit floating-point arithmetic; TensorFloat-32, an optional reduced-mantissa matrix-multiply mode. HP kernels use FP32 and disable TF32. Triton's `tf32x3` mode is also excluded. |
| AOTI | AOTInductor: the ahead-of-time deployment backend of TorchInductor, used here to package an exported energy/force graph as `.pt2`. It is distinct from the legacy TorchScript `.pt` route. |
| Reference route | The original differentiable implementation selected by `policy="reference"`. This selection does not imply FP64 arithmetic. |

Irrep degree/parity follows the [e3nn irreps convention](https://docs.e3nn.org/en/stable/api/o3/o3_irreps.html).
The native harmonics use component normalization: on unit directions, the
squared norm of a degree-`l` block is `2l+1`, as in
[e3nn component normalization](https://docs.e3nn.org/en/stable/api/o3/o3_sh.html).
The component coordinates can differ between angular bases even when this
normalization agrees.

Throughout this guide, `device="cuda"` and `torch.cuda` denote PyTorch's GPU
device API. HIP-enabled PyTorch uses these same names on K100; they do not
identify K100 as NVIDIA hardware. See [PyTorch HIP semantics](https://docs.pytorch.org/docs/2.7/notes/hip.html).
Backend-specific export support is described separately below.

## API overview

| API | Purpose | Input / output contract |
| --- | --- | --- |
| `EdgeTensorProduct.create(...)` | Create a standalone edge TP | Positive integer C, GPU FP32, nonnegative L, natural parity, fixed unit base coefficients |
| `EdgeTensorProduct(tp, ...)` | Wrap an existing ICTC TP | Preserves fixed coefficients, paths and external-gate conventions |
| `GraphMessage(tp, ...)` | Graph gather/TP/gate/scatter | Node/edge degree dictionaries, gates and repeated src/dst indices → node path blocks |
| `PackedGraphMessage(tp, ...)` / `.create(...)` | Native packed graph TP | Same layout as `EdgeTensorProduct(layout="ir_mul")`; 2D input/output buffers and caller-supplied topology |
| `direction_harmonics(n, L, policy=...)` | Cartesian direction harmonics | Unit directions `[E,3]` → list of `[E,2l+1]` blocks |
| `configure_symmetric_contraction(sc, ...)` | Enable a fast factorization of an existing SC | Selects a plan in place; forward retains the original module interface |
| `create_symmetric_contraction(...)` | Create a high-performance SC | Configurable C/L/correlation, existing full CG bridge-U; returns `ICTDBridgeUSymmetricContractionSO3` |
| `channel_linear(x, W, bias=None)` | Shared channel GEMM | `[*,Cin,m]` and `[Cout,Cin]` → `[*,Cout,m]` |
| `normalized_silu(x, scale=1.0)` | Fused scalar activation and derivatives | Computes `SiLU(x)*scale`; does not infer a model normalization constant |
| `create_model(...)` | Create an initialized model for examples or new training | Configurable channels/lmax/edge_lmax/interactions/correlation; returns an ordinary ICTC model |
| `check_model(model, ...)` | Check whole-model coverage without mutation | Returns `supported` and `reasons`; does not compile or modify the model |
| `configure_model(model, ...)` | Select a whole-model eager execution plan | Returns the selection, reasons, operator counts and layouts |
| `energy_and_forces(model, ...)` | Evaluate model energy and forces on an existing graph | Returns `energy[B]`, `forces[N,3]` and `atomic_energy[N]`; does not add E0 |
| `prepare_for_export(model, ...)` | Prepare whole-model FX/AOTI export | Registers or prepares independent Triton operators for export lowering; does not write an export file |

Importing the API does not trigger GPU compilation. The first actual call may
build constant plans and JIT-compile kernels; cold-start time must be reported
separately from warm throughput. Place generated source and compilation caches
inside the corresponding remote task directory.

HP GPU kernels use strict FP32, with no TF32 or Triton `tf32x3` execution option.
Whole-model configuration, standalone TP/graph messages, channel GEMM and SC
configuration set `torch.set_float32_matmul_precision("highest")` and disable
matmul/cuDNN `allow_tf32` for the process, without restoring the previous settings.
They do not cast tensors or change parameter trainability. Custom scripts,
exports and calls into other libraries should retain these three settings.
Existing compiled binaries must be re-exported to change their compile-time
arithmetic selection.

## 1. Standalone edge TP

Create a native ICTC operator:

```python
import torch
from ictc_o3 import EdgeTensorProduct, direction_harmonics

op = EdgeTensorProduct.create(
    channels=96, input_lmax=3, edge_lmax=3, output_lmax=3,
    layout="ir_mul", device="cuda",
)
E = 257
C = op.tp.channels
blocks = {
    l: torch.randn(E, C, 2*l+1, device="cuda", requires_grad=True)
    for l in range(4)
}
direction = torch.nn.functional.normalize(torch.randn(E, 3, device="cuda"), dim=-1)
harmonics = torch.cat(direction_harmonics(direction, 3, policy="generated"), dim=-1)
gates = torch.randn(E, op.gate_width, device="cuda", requires_grad=True)

features = op.pack_input(blocks)
output = op(features, harmonics, gates)
output_blocks = op.unpack_output(output)
loss = output.square().mean()
loss.backward()
```

All floating inputs must be FP32. If the process default dtype is float64,
add `dtype=torch.float32` when constructing inputs. The gates above are examples;
real applications should supply the current output of their radial network.

Stable shape conventions:

- `features`: `[E, C*(input_lmax+1)^2]`, with degrees ordered from 0 to L.
- `harmonics`: `[E, (edge_lmax+1)^2]`, with Cartesian degree blocks ordered from 0 to L.
- `gates`: `[E, P*C]` or `[E,P,C]`; **use `op.paths` as the path order**.
- `output`: `[E, op.output_width]`, ordered first by output degree, then by the original TP path slot.
- `unpack_output` returns `{l: [E, path_count_l*C, 2*l+1]}`, retaining path multiplicity.

`layout="ir_mul"` stores `[angular component,channel]` within each degree/path: the
per-degree output is `[E,path_count_l,2l+1,C]` before flattening.
`layout="mul_ir"` uses `[E,path_count_l,C,2l+1]`. Required input/output
conversions are part of the call. `pack_input` and `unpack_output` change memory
layout without rotating the angular basis.

The previous generated execution route remains an explicit option:

```python
op = EdgeTensorProduct.create(
    channels=64, input_lmax=3, edge_lmax=3,
    layout="mul_ir", execution="legacy",
)
```

`execution="legacy"` uses the original joint backward schedule and rematerializes
`mul_ir` input and output-gradient layout conversions inside the TP autograd
boundary; `ir_mul` still writes native buffers directly. It uses the current
Cartesian coefficient generator and second-derivative implementation, with the
same input, output, path and fixed-coefficient contracts. The default
`execution="direct"` specializes execution by the requested gradient roles.
The constructor that wraps an existing TP accepts this option too. It controls
standalone TP only; use `configure_model` for whole models.

New factories use native ICTC coefficient normalization and do not automatically
rescale paths for e3nn/cuEq. To extract an operator from an existing model, wrap
its original TP:

```python
op = EdgeTensorProduct(
    model.interactions[1].tp,
    input_lmax=model.lmax,
    edge_lmax=model.ictd_fix_edge_lmax,
    layout="ir_mul",
)
```

Set `input_lmax` from the actual layer inputs. The first layer may accept scalar
inputs only, so the model-wide L is not always appropriate. Existing TP base
coefficients must already be fixed by the architecture. The API rejects
trainable coefficients rather than freezing them. Non-unit fixed coefficients
retain their values; gradients remain available for gates and feature inputs.

This wrapper is intended for standalone calls and has a different signature
from the original model TP. Its saved state has a `tp.*` prefix. Whole-model
optimization uses `configure_model`, which does not replace modules or change
checkpoint keys.

Standalone edge operators write outputs directly, without node aggregation.
First-order backward, mixed second derivatives and existing higher-order autograd
routes remain available. This GPU JIT wrapper cannot be placed directly inside
`make_fx`, `torch.compile`, functorch transforms or autocast. Whole-model AOTI uses the prepared route
in section 6.

Use `torch.autograd.grad` for VJPs:

```python
vectors = torch.randn(E, 3, device="cuda", dtype=torch.float32, requires_grad=True)
Y = torch.cat(direction_harmonics(torch.nn.functional.normalize(vectors, dim=-1), 3), -1)
y = op(features, Y, gates)
dx, dr, dg = torch.autograd.grad(
    y, (features, vectors, gates), grad_outputs=torch.randn_like(y), create_graph=True,
)
```

`create_graph=True` retains the graph needed to differentiate force or geometry
gradients with respect to parameters. This example preserves the chain from
direction harmonics to the original edge vectors. Position and cell chains are
provided by the caller's graph or the whole-model helper.

## 2. Graph message aggregation

Use `GraphMessage` when node aggregation is required:

```python
from ictc_o3 import GraphMessage

message = GraphMessage(
    tp, input_lmax=3, edge_lmax=3,
    layout="packed", edge_chunk_size=65536, reduction="auto",
)
# node_blocks[l]: [N,C,2l+1], C=tp.channels
# edge_blocks[l]: [E,1,2l+1]
# radial_gates: [E,tp.num_paths*C], current radial MLP output
# src/dst: [E] int64 CUDA, valid indices into the same N-node buffer
node_paths = message(node_blocks, edge_blocks, radial_gates, src, dst, mask=None)
```

Source and destination indices may repeat; graph reduction accumulates into
the correct nodes. Path blocks remain separate so a subsequent equivariant
linear layer can mix them as in the original model. This API evaluates neither
the radial MLP nor the neighbor list. An optional `mask` is fixed FP32 `[E]` or
`[E,1]`; trainable masks require the original reference implementation.
Layout/chunking options belong to the wrapper and do not modify execution
settings or parameters on the shared TP.

`reduction="auto"` enables GPU CSR forward for C>=96, all degrees<=3,
E>=4096 and at least 16 incoming edges per node on average. Other graphs retain
atomic forward. `reduction="atomic"` selects the original schedule;
`reduction="csr"` requires CSR forward with L<=3. CSR plans the current topology
on the GPU, preserves the original edge array order and does not copy the whole
gate tensor. Warps accumulate neighbors in registers before writing nodes.
First-order backward and mixed second derivatives retain the existing analytic kernels.
FP32 reduction order may change, so results are not guaranteed to be bitwise identical.

Edge TP `ir_mul/mul_ir` and graph/model `packed/fused_pack` operate at different
layout levels and are not interchangeable. `fused_pack` fuses degree copies at
graph-kernel boundaries while retaining gather/scatter, initialization
and graph reduction. The default is `packed`.

## 3. Other high-performance operators

```python
from ictc_o3 import (
    direction_harmonics, create_symmetric_contraction, configure_symmetric_contraction,
    channel_linear, normalized_silu,
)

Y = direction_harmonics(unit_directions, 3, policy="generated")

sc = create_symmetric_contraction(num_elements=3, channels=32, lmax=2,
                                  target_lmax=2, correlation=2)
# Or keep an existing trained SC: sc = model.products[layer_index]
info = configure_symmetric_contraction(sc, policy="generated")
# sc: ICTDBridgeUSymmetricContractionSO3, GPU FP32/full CG bridge-U
# x: [N, C*(L+1)^2], ICTC mul_ir degree blocks
# node_attrs: [N,num_elements] fixed one-hot, species_idx: [N] int64
value = sc(x, node_attrs, node_type_idx=species_idx)

mapped = channel_linear(block, W)  # block [N,Cin,2l+1], W [Cout,Cin]
activated_scalar = normalized_silu(scalar_features, scale=model_activation_scale)
```

SC preserves the original weights, parameter identities and checkpoint keys.
Factorization requires discrete `node_type_idx`. Calls with only `node_attrs`
retain the original module behavior and do not replace differentiable species
inputs with `argmax`. Kernel planning still checks reconstruction of the saved U.
SC retains the validated internal e3nn basis bridge; it does not regenerate
higher-order U entirely from Cartesian algebra.

Channel GEMM supports ordinary autograd, including weight gradients. The API
allows bias only when `m=1`; the caller must also ensure that the block is an
even scalar (`0e`) if O(3) equivariance is required. Apply SiLU directly only to
invariant features such as `0e` scalars. SiLU on odd scalars (`0o`), or
componentwise on nonscalar blocks, generally breaks O(3) equivariance. `scale` is a fixed
number; the API does not infer the model's normalization constant.

For direction harmonics and SC, `policy="auto"` preserves reference fallbacks,
explicit `generated` requires coverage, and `reference` selects the reference
route. Differentiable CPU fallbacks remain available for GEMM/SiLU.

## 4. Whole-model construction and force evaluation

The model factory constructs an initialized network. It does not load a trained
potential or automatically select an HP plan:

```python
from mace_ictc.hp import create_model, check_model, configure_model, energy_and_forces

model = create_model(
    channels=96, lmax=2, edge_lmax=2, num_interactions=4, correlation=2,
    atomic_numbers=(1,6,8), cutoff=5.0,
    avg_num_neighbors=50.0, device="cuda", dtype=torch.float32,
)
print(check_model(model, training=True))
```

Factory parameters are configurable; defaults are C64/L3, two interactions and correlation=3.
New TP base coefficients are fixed by construction, while radial, linear and
SC parameters are trainable. Existing checkpoint trainability is not modified.
`channels` and `num_interactions` accept positive integers; `lmax` is
nonnegative. `edge_lmax` defaults to `lmax` and must be at least as large.
Fast kernels are generated for the actual C, paths and angular widths, with
load/store masks for channel tails that are not multiples of 32.

`correlation` supports 2 and 3, with compatibility for 1. The factory uses the
existing full CG baseline bridge-U. `use_reduced_cg=True` raises an explicit error
rather than enabling cuEquivariance (cuEq) basis generation. The process default dtype is
preserved. For dynamic AOTI, the factory builds e3nn submodules through FX and
restores the previous e3nn optimization defaults afterwards.

Interaction depth and angular degree are not fixed to two layers/L<=3.
Structural support remains subject to native model basis construction and
device resources. Unvalidated high-degree/wide combinations are not validated
performance configurations.

Force inference:

```python
model.eval().requires_grad_(False)  # Explicit inference choice, not training.
info = configure_model(model, training=False, policy="generated", layout="packed")
with torch.no_grad():
    result = energy_and_forces(model, pos, Z, batch, src, dst, shifts, cell)
energy = result["energy"]          # [B]
forces = result["forces"]          # [N,3]
```

`Z`, `batch`, `src` and `dst` are int64 tensors on the same device; `pos`,
`shifts` and `cell` are FP32. `shifts[E,3]` contains lattice-coordinate shifts,
and `cell[B,3,3]` contains each graph's cell. Geometry and neighbor lists must
already be prepared. This helper does not add fitted E0, rebuild graphs or
perform MPI/MD.

Energy is the sum of the original model's atomic forward outputs. If the core
returns interaction energy only, the API returns interaction energy only.
For absolute energy, retain the checkpoint/E0 wrapper or add the same fitted
E0 table. Training labels must use a consistent energy convention.

The helper enables the autograd needed for forces even inside `no_grad`, then
detaches inference results. `torch.inference_mode()` is rejected because
coordinate derivatives are required. An optional fixed `owned_mask[N]`
restricts the differentiated energy sum to owned atoms while returning forces
for all local/ghost coordinates. The LAMMPS interface remains responsible for
reverse ghost communication.

Existing checkpoints:

```python
from mace_ictc.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF

potential = LAMMPS_MLIAP_MFF.from_checkpoint(
    "trained_model.pth", element_types=["H","C","O"], device="cuda",
    avg_num_neighbors=training_average_neighbors,
    operator_plan="generated", generated_layout="packed",
)
model = potential.wrapper.model
```

Species, training average neighbor count, cutoff, dtype and E0 must come from
the actual checkpoint/training record. The API does not force FP64 checkpoints
to FP32 or silently freeze trainable TP coefficients.

## 5. Training

Configure the original model with its intended parameter trainability flags:

```python
configure_model(model, training=True, policy="generated", layout="packed")
model.train()
parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(parameters, lr=1e-3)

optimizer.zero_grad(set_to_none=True)
out = energy_and_forces(model, pos, Z, batch, src, dst, shifts, cell, training=True)
loss = (out["energy"] - target_energy).square().mean() \
     + (out["forces"] - target_forces).square().mean()
loss.backward(inputs=parameters)
optimizer.step()
```

`training=True` retains the force graph and mixed second derivatives. It does
not call `model.train()` or change parameter trainability. When resuming
training from an inference model, restore the original checkpoint flags:
`model.requires_grad_(True)` would also make fixed base coefficients trainable.
If loss gradients with respect to leaf positions are required, set
`pos.requires_grad_(True)` before the call and request the intended gradient
targets. Ordinary parameter training can use the parameter-only backward above.

On K100 (`gfx926`), FP32 CG contraction kernels use native HIP for feature gradients,
SC-weight gradients and mixed second derivatives. Species banks are
indexed directly by node type; weight gradients use reductions over blocks
of 256 nodes. Generated-message second backward with fixed TP coefficients
accumulates directly into harmonic inputs and supports coordinate gradients
requested by ordinary `loss.backward()`.

These routes retain the original node/channel and edge/channel layouts and use
native HIP kernels for these backward operations. Third and higher derivatives
retain differentiable expressions. Coverage is eager FP32 with fixed TP
coefficients, differentiable gates and trainable SC weights.
`training_gate_schedule="materialized"` uses the complete second-derivative route.

`ForceTrainer` and the CLI expose layout selection too:

```python
from mace_ictc.training import ForceTrainer

trainer = ForceTrainer(
    model, train_loader, config=training_config, device="cuda", dtype=torch.float32,
    operator_plan="generated", generated_layout="fused_pack",
    backward_targets="parameters",  # other established trainer arguments unchanged
)
```

Example training command using preprocessed data; preprocessing must also run
on the designated remote compute resources:

```bash
mff-train --data-dir /task/data --dtype float32 \
  --channels 64 --lmax 3 --max-ell 3 --correlation 3 --num-interaction 3 \
  --route baseline --product-backend ictd-bridge-u --angular-basis ictd \
  --freeze-conv-tp-weight --operator-plan generated --generated-layout packed \
  --backward-targets parameters
```

`--freeze-conv-tp-weight` is an explicit architecture choice for a new model.
It changes an existing checkpoint whose TP coefficients are trainable.
`--operator-plan auto` does not change dtype or freeze weights. Explicit make_fx
training capture uses the reference route under `auto`; its behavior and
performance must be reported separately from generated eager training.

## 6. AOTI / LAMMPS

Python preparation entry point:

```python
from mace_ictc.hp import prepare_for_export

model.eval().requires_grad_(False)
info = prepare_for_export(model, policy="generated", layout="fused_pack")
# Continue with the existing force make_fx -> torch.export -> AOTInductor pipeline.
# Optional preparation-only control: retain canonical U on its original device.
# info = prepare_for_export(model, policy="generated", reference_buffer_policy="device")
```

`layout=None` retains an explicitly selected layout; a new model defaults to
packed. `dynamic=True` is the default. Existing TorchScript e3nn submodules
would bake the example N into a symbolic trace, so the API reports their names
before export. Set `e3nn.set_optimization_defaults(jit_script_fx=False)` before
rebuilding/loading the model. Pass `dynamic=False` only when deliberately
preparing a fixed-N package. The export CLI sets this option before model loading.

The function prepares the model. Use the existing export CLI to create the
package, validate it and write LAMMPS metadata:

```bash
mff-export-aoti --checkpoint trained_model.pth --elements H,C,O \
  --device cuda --operator-plan generated --generated-layout packed \
  --avg-num-neighbors TRAINING_VALUE --embed-e0 --out /task/model.pt2
```

The CLI uses the checkpoint weights, actual architecture and precision.
`--generated-layout fused_pack` compiles fused copies into the new package.
Changing Python configuration does not update an existing `.pt2` file.
Export preserves the owned-energy interface, dynamic N/E and message-passing
depth. The CLI-generated `.meta` also records the requested layout. For complete
build and usage instructions for `pair_style mff/torch` / `mff/torch/kk`, see the
[interface documentation](../lammps_user_mfftorch/README.md).

High-L inference preparation uses bounded export implementations. L4/5, correlation=3
SC uses a shared quadratic prefix, sparse final coupling and analytic first
backward kernels. Correlation=2 with channel tails and nonscalar targets also uses sparse export.
High-L TP is partitioned by input degree and at most two paths; C>=32 adds
32-channel tiling, retaining graph aggregation and geometry derivatives.

The default `reference_buffer_policy="auto"` leaves original U on the CPU
after factorization for validated single-device correlation=3 models. `"device"`
changes preparation-time residency of the original buffers without changing
exported operators. The CLI option is `--reference-buffer-policy auto|device`.
These constants are not transferred at every step.

Dense U remains registered with its original checkpoint keys. Only the
factorization tables, projections and frozen weights actually referenced by
the force graph enter the export package. Parameter or U changes after
preparation require preparation/export again; old frozen banks are no longer
valid. A standalone edge wrapper that writes outputs directly cannot replace
graph messages in the export graph, which require node reduction.

## 7. Coverage, defaults and fallbacks

Whole-model automatic coverage requires GPU FP32, internal FP32, positive
integer channels, nonnegative hidden/edge L with edge L>=hidden L, positive
interaction depth, Cartesian ICTC, correlation=2/3 with compatibility for 1, the
existing full CG baseline bridge-U, no interaction attention and no cuEq mixing.
Inference requires already frozen parameters; training requires fixed
architecture-level TP base coefficients. Radial, linear and SC weights retain
their original trainability. API coverage and correctness validation do not
guarantee a speedup for every graph, layer count or configuration.

### Unified policy entry point

One entry point enables the applicable validated HP strategies:

```python
# Existing GPU FP32 model; preserve its parameters and checkpoint keys.
info = configure_model(model, training=True, policy="generated")
print(info["selected"], info["counts"])
```

`wide_channel_policy` and `high_l_policy` both default to `auto`. `generated`
requires coverage and raises an explicit error when its conditions are not met;
`auto` may select the reference route and return the reasons. Neither freezes
weights. For inference, explicitly freeze parameters, pass `training=False`
and use the complete force evaluation in section 4.

| Component | Rules used by the unified policy |
| --- | --- |
| TP, messages and radial network | Generated messages, differentiable gates and radial recomputation, retaining gather/scatter and force derivatives |
| SC | Circuits for correlation=2; degree partitioning for L4/5 training and L5 inference with channel tails; bounded sparse coupling for L4/5 correlation=3 |
| Channel and species matrices | Shared channel GEMM in inference; shared species GEMM for C>=96, hidden/edge L<=3 and at most four species |
| Node aggregation | CSR forward for C>=96, hidden/edge L<=3, E>=4096 and average indegree>=16 |
| High-L TP | Streamed geometry for L4/5 and C>=64; channel tiling depends on training/inference and device, as described below |
| Precision and parameters | Strict FP32 with TF32 disabled; trainable SC, radial and linear weights participate in evaluation and differentiation |

These conditions form one automatic policy set. Different SC circuits and
aggregation kernels are mutually exclusive schedules; the entry point selects
the applicable route. Enabling the set does not guarantee simultaneous latency
and memory reductions for every architecture.

For C>=96 and hidden/edge L<=3, two whole-model optimizations are enabled by
default. With at most four species, shared GEMM by species avoids `[N,C,C]`
weight copies and their gradient scatter. Large, sufficiently dense graphs use
the CSR forward described above. Species banks, radial, channel and SC weights
remain trainable. Lower channel counts retain the preceding HP schedules;
interaction depth and correlation=2/3 remain independent choices.
`wide_channel_policy="off"` disables these two wide-channel optimizations while
retaining the other HP operators:

```python
configure_model(model, training=True, policy="generated", wide_channel_policy="auto")
# Restore the previous HP schedule for a controlled comparison.
configure_model(model, training=True, policy="generated", wide_channel_policy="off")
```

Explicit `species_linear_schedule="grouped"` groups element-conditioned linear
maps with integer species indices, including `self_connection`, by the actual
species present. Each nonempty group uses one GEMM, avoiding temporary
`[N,C_out,C_in]` weights. The algorithm accepts configurable C, L, species count
and interaction depth, using actual tensor shapes. Parameter and higher
gradients retain ordinary PyTorch autograd.

Node groups are rebuilt for each call and reused across angular blocks within
that call. Membership that changes with ghost/owner topology is not cached.
Memory overhead scales with node features.

```python
configure_model(model, policy="generated", species_linear_schedule="grouped")
# Controlled baseline using per-node weights.
configure_model(model, policy="generated", species_linear_schedule="gather")
```

This option is eager-only: group counts require a small GPU-to-CPU metadata
synchronization, so capture is unsupported. Small systems may not become faster.
`auto` retains the defaults in the table, and reference mode restores the
reference route. This setting is independent of SC `species_schedule`.
Soft species mixtures through `forward(node_attrs)` retain the original
formula. `check_model` still determines whole-model HP coverage.

Large-system MD also uses memory outside PyTorch's caching allocator, including
LAMMPS buffers and communication buffers. Account for these allocations when
choosing a gate-cache budget; the Python HP settings do not cap total process
GPU memory.

For high-L models, `high_l_policy="auto"` selects L4/5 schedules. Cor3 SC uses
a shared quadratic CG prefix and sparse final coupling to avoid an oversized
monolithic cubic circuit. Cor2 keeps its circuit, partitioning nonscalar outputs
by degree during training and during L5 inference with non-multiple-of-32
channels to bound compilation complexity.

High-L TP with C>=64 generates geometry by input/output angular component.
L4 training uses 32-channel tiles, while inference retains the original tiling.
L5 correlation=2 training retains the original tiling and inference uses 32-channel tiles.
L5 correlation=3 training uses 32-channel tiles; inference uses the original tiling on
NVIDIA GPUs with compute capability 8.9 and 32-channel tiles elsewhere. This
device-dependent schedule does not guarantee a speedup for every graph.

Cor3 sparse execution sizes work blocks by coefficient count per row, bounding
per-thread intermediates when a row has many coefficients. Original SC learned weights enter
evaluation and differentiation through differentiable projections. Training weights are
not cached, coefficients are not truncated, and model parameterization is
preserved. TP and public features retain the Cartesian basis; SC retains the
existing bridge-U and independent CG-prefix implementation.

```python
model = create_model(channels=128, lmax=5, correlation=3, num_interactions=3)
info = configure_model(model, training=True, policy="generated", high_l_policy="auto")
print(info["high_l_active"], info["counts"])
# Restore the preceding high-L schedules without replacing weights/checkpoints.
configure_model(model, training=True, policy="generated", high_l_policy="off")
```

Standalone SC uses the same policy. For force-loss mixed backward,
`configure_symmetric_contraction(sc, training=True)` selects bounded training
circuits. Pass `configure_symmetric_contraction(sc, high_l_policy="off")` to
restore the previous monolithic circuit. L<=3 default schedules are unchanged.
High-degree coverage does not imply a performance gain for every channel count,
graph size or GPU. Small-graph launch costs, initial constant planning and JIT
must be considered separately.

For single-device L4/5 correlation=3, original dense U and derived transpose buffers move
to the CPU by default after the factorization plan passes validation. Warm GPU
execution uses the factorized coefficients, while features, gates, learned
weights and backward computation remain GPU FP32. Only fixed constants move;
activations and training weights are not transferred at every step.

The original buffers remain registered with identical checkpoint keys, shapes,
dtypes and values, so `named_buffers()` may contain both CPU and GPU tensors.
Reference calls and device-residency selection restore the buffers to their
original device. Differentiable `node_attrs` retain the reference route.
High-L generated export preparation retains CPU residency; checkpoint/fallback
export preparation restores device buffers.

```python
# Same default one-call HP strategy, including validated high-L memory savings.
configure_model(model, training=True, policy="generated")
# Retain conventional device buffers for external tooling or a control run.
configure_model(model, training=True, policy="generated", reference_buffer_policy="device")
# The same residency option is available for standalone SC.
configure_symmetric_contraction(sc, training=True, reference_buffer_policy="device")
```

Initialized distributed process groups and DistributedDataParallel (DDP) or DataParallel wrappers
automatically retain device buffers. Configure HP after wrapping. Before
creating a parallel wrapper around a model that already has CPU constants,
select `device`. Frequent alternation between discrete-species HP calls and
differentiable-attribute reference calls transfers constants; such workloads
can keep `device` residency.

High-L generated export defaults to CPU U; use
`prepare_for_export(..., reference_buffer_policy="device")` for device residency.
Measure peak memory separately for prepared Python, standalone `.pt2` runners
and actual LAMMPS processes. Eager constant-byte savings cannot directly predict
the peak-memory difference in the other execution modes.

The GPU CSR schedule applies to eager training/inference. AOTI retains
prepared Triton message kernels, with separate bounded high-L SC/TP
implementations from section 6. Measure complete forces with the actual
AOTI/LAMMPS package; eager speedups do not transfer automatically.
Explicit `generated` retains reconstruction-accuracy checks against the actual
U tensors. Exceeding the reconstruction budget raises an error rather than
reporting an original-SC fallback as HP support. Initial planning/JIT is
separate from warm throughput.

```python
# Keep the established default; auto reports unsupported-model reasons.
info = configure_model(model, training=True, policy="auto", layout="packed")
# Opt in to fused copies on a validated model.
info = configure_model(model, training=True, policy="generated", layout="fused_pack")
# Revert only the HP layout.
configure_model(model, training=True, policy="generated", layout="packed")
# Revert the eager operator recipe.
configure_model(model, training=True, policy="reference")
```

The training CLI selects the reference route with `--operator-plan reference`.
Checkpoint/export paths can retain their original route with
`operator_plan="checkpoint"` / `--operator-plan checkpoint`.
`prepare_for_export(..., policy="checkpoint")` retains the current checkpoint
route but does not undo an already prepared deployment callable. Reload the
original checkpoint when a clean original route is required.

`fused_pack` remains opt-in; its speed and memory effects depend on graph,
model and execution mode. The default is `packed`.

## 8. Advanced combinations requiring separate validation

Cor2 configurations using reduced CG are outside current `auto` coverage.
The existing low-level selector
`mace_ictc.models.operator_plans.set_mace_backbone_plan` remains available for
explicit combinations. For an FP32/C64 model independently validated by the
caller:

```python
from mace_ictc.models.operator_plans import set_mace_backbone_plan

set_mace_backbone_plan(
    model, bond_first=True, angular_backend="triton_full", path_assembly="concat",
    edge_fusion="triton_remat", activation_fusion=True, compact_contraction=True,
    factorized_circuit=True, streamed_gates=True, generated_message=True,
    generated_second=True, fused_harmonics=True, fold_frozen_path_weights=True,
    shared_channel_gemm=False, edge_chunk_size=65536, gate_cache_bytes=0,
    generated_layout="packed",
)
```

This selector returns registered-module counts and does not guarantee fast-kernel
dispatch for arbitrary architectures or inputs. Preserve the corresponding
numerical validation; bypassing high-level `generated` coverage/trainability
checks does not establish automatic support. Validation should cover FP32
energy/forces, per-atom energy predictions with a consistent E0 convention,
every force-loss parameter gradient and reconstruction of the actual saved U.
Module counts cannot establish
correctness.

## Low-degree derivatives and graph message schedules

For L<=3, standalone `EdgeTensorProduct` selects kernels by the first/second
gradient roles actually requested. The edge second-derivative route retains
native packed buffers and writes directly, avoiding graph-aggregation layout
conversions and atomic additions. Geometry second derivatives reduce directly
into harmonic inputs; higher derivatives retain a native-expression fallback.
The API still does not freeze existing trainable TP base coefficients. Learned
radial gates remain differentiable functions of geometry.

Whole-model `message_schedule` defaults to `"native"`, retaining the existing
HP schedule:

```python
# Configure existing models/checkpoints without changing weights or architecture.
configure_model(model, training=True, policy="generated", layout="fused_pack",
                message_schedule="roles")
configure_model(model, training=False, policy="generated", layout="fused_pack",
                message_schedule="roles")

# Restore the original message schedule; other SC/high-L/wide policies stay configurable.
configure_model(model, training=True, policy="generated", message_schedule="native")
```

`"roles"` uses the original Cartesian basis and native GPU backward kernels compiled
for the requested gradient inputs. The retired `"compact"` coordinate-switching schedule is
unsupported. Use `"roles"` or `"native"` to retain the native Cartesian basis.
This restriction does not affect standalone TP `execution="legacy"` layout
compatibility.

Explicit role-based message scheduling covers hidden/edge L<=3. Higher degrees
retain the `native` route, `high_l_policy` and AOTI. In particular, `canonical`
memory access and experimental multi-warp/grouped schedules are not default
recommended configurations. Channels, interaction depth and correlation remain
caller-selected architecture parameters. Performance validation depends on
the actual device and shapes.

The native packed graph interface accepts existing 2D buffers:

```python
from ictc_o3 import PackedGraphMessage

op = PackedGraphMessage.create(channels=64, input_lmax=3, edge_lmax=3,
                               cartesian_schedule="native", reduction="csr")
# x: [N, C*(L_in+1)**2], angular component before channel within each degree
# sh: [E, (L_edge+1)**2]; g: [E, op.gate_width]
# src/dst: [E], int64 CUDA, valid node indices; repeated/unsorted entries allowed
# Output: [N, op.output_width], separate paths, same layout as EdgeTensorProduct.ir_mul
out = op(x, sh, g, src, dst)
grad_x, grad_g = torch.autograd.grad(out, (x, g), output_cotangent)
```

Use `PackedGraphMessage(tp, input_lmax=..., edge_lmax=...)` to wrap an existing
TP with fixed coefficients. `create` constructs new fixed unit base coefficients.
`cartesian_schedule="native"` uses the original coordinates and is the only
supported coordinate schedule. This interface supports ordinary and mixed
autograd in eager GPU FP32 and is not a direct AOTI entry point. The dictionary
`GraphMessage` interface remains available for `[N,C,2l+1]` feature blocks.
Edge-only throughput does not represent complete energy/force throughput.

## Device-side CSR for large graphs

Whole models can explicitly select CSR forward and device-side grouping while
retaining existing first-order and mixed second-order backward schedules.
These are tuning options; their effect on complete energy/force evaluation
depends on C, L, graph topology and GPU.

```python
configure_model(model, training=False, policy="generated", layout="fused_pack",
                message_schedule="roles", graph_reduction="csr",
                topology_backend="device", csr_accumulator_limit=24)
configure_model(model, training=True, policy="generated", layout="fused_pack",
                message_schedule="roles", graph_reduction="csr",
                topology_backend="device", csr_accumulator_limit=24)

op = PackedGraphMessage.create(channels=64, input_lmax=3, edge_lmax=3,
     cartesian_schedule="native", reduction="csr", topology_backend="device",
     csr_accumulator_limit=24)
out = op(x, sh, g, src, dst)
```

`graph_reduction="auto"` retains the existing wide-channel/high-L policy.
`"atomic"` and `"csr"` explicitly select forward reduction.
`topology_backend="native"` uses stable sorting by default. `"device"`
recounts, builds pointers and groups indices on the GPU for every call,
supporting unsorted and repeated indices without caching graphs, features or
learned weights. Sorted input is detected on the GPU and keeps its edge order.
For unsorted input, within-node edge order is not guaranteed to be stable;
floating reduction order can change, so validation uses numerical tolerances.
`csr_accumulator_limit=None` retains the original tiling policy. A positive
integer controls the CSR accumulator budget. Reference mode does not use this
device fast path.

These options preserve architecture, dtype and parameter trainability.
Validated coverage remains eager GPU FP32; the CSR schedule does not expand
AOTI performance validation. Experiments with source-fused backward,
cross-degree fusion and persistent warp scheduling did not show consistent
benefits and are not defaults.

## Native-basis implementation boundaries

Forward, backward and training state retain the established ICTC angular basis
and CG/SC conventions. Direct edge second-derivative stores, gradient-role
compilation, native packed buffers, chunked radial gates and existing
high-L/wide-channel schedules are retained. `compact_contraction` names the
existing SC compression strategy, not the retired coordinate switch.

After loading a model with old experimental flags,
`configure_model(..., message_schedule="roles")` clears those flags and temporary
coordinate caches while preserving weights, architecture, dtype and trainability.
Explicit requests for the old `compact` route raise an error rather than
silently substituting a route and reusing obsolete performance claims.

Native-basis CSR chunking can explicitly enable `csr_slice_search=True`.
When original edge IDs are monotonic within each node's stable group, lower/upper
bound searches restrict neighbor traversal to the current chunk; nodes with no
edges in that chunk finish immediately. Device grouping uses this search only
when the GPU confirms sorted input, otherwise retaining a full scan so unsorted
edge IDs are not treated as ordered. CG, coordinates, paths and differentiation
formulas are unchanged. The option defaults off and needs performance validation
for the actual shapes.

## Optional SC species weights indexed by node

`configure_model(..., species_schedule="indexed")` and
`configure_symmetric_contraction(..., species_schedule="indexed")` select a
node-indexed species-coefficient route for native full CG circuits with L<=3.
It retains the CG polynomial and trainable weight bank while eliminating per-species
`nonzero`, node grouping and full-buffer scatter-back. Input features, species
groups and learned parameters are not cached. First, parameter and mixed second
gradients remain available.

The default `species_schedule="grouped"` retains the original route.
High-L and existing partitioned/sparse SC retain their plans. Returned
`indexed_species_circuits` counts, or standalone SC `indexed_species_active`,
report actual coverage. `policy="reference"` clears indexed execution.
Discrete `node_type_idx` must be int64 `[N]` on the same device, with every
index inside the species-bank range. Calls with only `node_attrs` retain the
original convention; differentiable species attributes are not replaced by
discrete indices.

This setting preserves radial paging and gate-cache budgets. Select capacity
and throughput configurations for the workload and check actual energy, forces,
parameter gradients, warm throughput and peak memory. Fixed-shape CUDA Graphs
require a separate caller-managed preparation and validation protocol; enabling
this option does not capture or cache whole-model calls.

## Native geometry schedules and edge chunks

Generated CG tables use the native sparse TP numerical-zero rule of `1e-12`.
Removed L3 construction residuals are approximately `1e-16`. The basis,
reference projection matrices and learned weights are preserved.

`geometry_schedule="auto"` retains the current plan. `"streamed"` computes
native geometry by input component during first-order backward, reducing the
intermediates that must remain in registers at the same time. `"native"`
restores the previous geometry evaluation order.
Measure performance for the actual C/L/device combination.

Standalone graph operators that already hold all gates can use larger chunks:

```python
op = PackedGraphMessage.create(
    channels=64, input_lmax=3, edge_lmax=3,
    reduction="csr", topology_backend="device", csr_accumulator_limit=24,
    csr_slice_search=True, geometry_schedule="streamed",
    edge_chunk_size=1_048_576,
)
y = op(x, sh, gates, src, dst)
```

Whole-model training can explicitly use the same geometry and chunk settings:

```python
configure_model(
    model, training=True, policy="generated", layout="packed",
    message_schedule="roles", graph_reduction="csr", topology_backend="device",
    csr_accumulator_limit=24, csr_slice_search=True,
    geometry_schedule="streamed", message_chunk_edges=1_048_576,
)
```

Capacity-oriented force inference retains smaller radial-gate chunks:

```python
model.eval().requires_grad_(False)  # Caller explicitly freezes inference parameters.
configure_model(
    model, training=False, policy="generated", layout="packed",
    message_schedule="roles", graph_reduction="csr", topology_backend="device",
    csr_accumulator_limit=24, csr_slice_search=True,
    geometry_schedule="streamed", message_chunk_edges=65_536,
    gate_cache_bytes=0,
)
```

`message_chunk_edges=None` retains current chunk sizes; a positive integer
overrides every layer. When training or standalone graph calls already hold
full gates, larger chunks reduce launch overhead. In streamed radial inference,
larger chunks enlarge temporary gate buffers, so measure peak memory too.
Capacity-oriented inference does not automatically enlarge edge chunks.
These eager-schedule performance observations have not been extended to AOTI
or existing deployment packages.

### Forward/backward chunks for large C64/L2 systems on K100

For a frozen two-interaction C64 model with hidden/edge L2 and full CG correlation=3,
large-system force evaluation on K100/DTK 26.04 can explicitly use larger forward
chunks and independent smaller backward chunks:

```python
model.eval().requires_grad_(False)
configure_model(
    model, training=False, policy="generated", layout="fused_pack",
    message_schedule="roles", species_schedule="indexed",
    graph_reduction="csr", topology_backend="native",
    csr_accumulator_limit=24, csr_slice_search=True,
    geometry_schedule="streamed", message_chunk_edges=1_048_576,
    gate_cache_bytes=0, inference_message_layout="degree",
    inference_gate_adjoint_rows=16_384,
    inference_message_adjoint_edges=32_768,
    inference_gemm_policy="dtk2604", inference_basis_fusion=True,
)
```

Each CSR forward chunk launches work for all destination nodes before excluding
nodes outside that chunk. Larger forward chunks reduce repeated searches and
launches; independent backward chunks bound force intermediates. This recipe
uses FP32 with TF32 disabled, preserves weights and CG/SC formulas, and keeps
gate caching disabled.

Choose both chunk sizes for the actual model, graph and device budget, and
validate memory use separately.

The inference options in this example control different operations:

| Option | Meaning |
| --- | --- |
| `message_chunk_edges` | Edges per TP forward chunk. Also determines backward chunks unless overridden. |
| `inference_message_adjoint_edges` | Edges per TP backward chunk with frozen parameters. `0` retains the forward chunk size; absent cached gates are recomputed for each backward chunk. |
| `inference_gate_adjoint_rows` | Maximum rows per radial gate-gradient GEMM with frozen parameters. `0` retains the original GEMM. This does not set a cache budget. |
| `inference_gemm_policy="dtk2604"` | Selects specialized FP32 gate GEMMs for supported K100/DTK 26.04 shapes; other shapes/stacks retain PyTorch operations. |
| `inference_basis_fusion=True` | Fuses basis conversion and packing around indexed low-degree SC kernels on K100. A feature tensor is retained for differentiable fallbacks, so measure its memory cost. |

These options do not freeze parameters and do not remove the gates' geometry
dependence. The chunk, GEMM and basis-fusion settings are independent of
`gate_cache_bytes`.

## Optional training gate memory schedules and mixed-backward tiling

`configure_model(..., training=True, training_gate_schedule=...)` changes how
gate intermediates are saved, using the original Cartesian/full CG message
formula:

| Value | Storage and recomputation behavior |
| --- | --- |
| `"materialized"` | Default; retains the established training route. |
| `"cached"` | Saves gates from the current forward; first backward retains narrow radial-feature gradients, and mixed second backward recomputes gate gradients in chunks. |
| `"recompute"` | Also recomputes forward gates in chunks during backward. |

The two optional routes trade recomputation and extra matrix multiplication
for lower peak memory; a speedup is not guaranteed. `message_chunk_edges`
controls their edge chunk size. `gate_cache_bytes` still controls only the
force-inference cache with frozen parameters and must be zero during training.
The `cached` training schedule saves all forward gates for the current autograd graph,
independently of that inference cache budget. It does not cache activations
or weights across calls.

```python
info = configure_model(
    model, training=True, policy="generated", layout="fused_pack",
    message_schedule="roles", species_schedule="indexed",
    graph_reduction="csr", topology_backend="device",
    csr_accumulator_limit=24, csr_slice_search=True,
    geometry_schedule="streamed", message_chunk_edges=65_536,
    training_gate_schedule="cached",  # Lower memory, with additional training time.
)
```

Coverage requires eager GPU FP32, hidden/edge L<=3, the original fixed TP
coefficients and trainable radial-network parameters. Parameter objects,
trainability, checkpoint keys and the CG/SC polynomials are preserved.
Fast mixed backward targets parameter-gradient requests such as
`autograd.grad(loss, trainable_parameters)`. Backward involving coordinate
Hessians and higher derivatives retains differentiable reference expressions,
without promising the same memory savings. These schedules are not used for
AOTI or inference acceleration.

Reconfiguring without `training_gate_schedule`, or switching to
`policy="reference"`, restores `"materialized"`. Returned information includes
the selected schedule and `counts["paged_training_gates"]`. The API does not
automatically enable the two schedules that add recomputation.

`mixed_channel_tile=64` or `32` changes only the channel tiling of generated
mixed second-derivative TP kernels. It does not change first-backward or
forward tiling and does not add gate recomputation. The default `None` retains
the established schedule; `0` explicitly selects one full-channel tile.
This option covers training L<=3 and can be combined independently with
`training_gate_schedule`. Wide channels may benefit; C64 benefits are not
guaranteed. One shape's speedup does not establish performance for all channel
counts. Omitting the option when reconfiguring clears its previous value.

### Operator layouts for force training on K100

`training_adjoint_channel_tile=32` or `64` independently changes the channel
layout of first TP backward, covering both `create_graph=True` force evaluation
and VJPs inside loss backward. Forward and mixed second derivatives
retain their own layout selections. The default `None` retains the existing
schedule; `0` uses full-channel unrolling. This option requires materialized
gates, eager execution and hidden/edge L<=3. Measurements on K100 C128/256, L3
favored 64-channel first backward with 32-channel mixed second backward.

`mixed_geometry_schedule="split"` moves the TP harmonic-input part of second backward into
a separate edge/channel kernel, using the existing 32-lane channel reduction
and direct harmonic-gradient stores. This reduces register pressure in the
feature, gate and output-cotangent kernel. The default `"fused"` retains the
joint kernel. Periodic C128/256, L3 graphs favored `"split"`; C64/L2 favored
`mixed_channel_tile=32` with `"fused"`. Choose for the actual C/L and graph size.

`training_gate_gemm_policy="k100_tiled"` uses 16384-row feature- and
weight-gradient GEMMs for the validated final radial projection, retaining the
original forward linear map. First-order and mixed second derivatives cover trainable
weights and inputs, including coordinate gradients from standard
`loss.backward()`. Conditions are gfx926, PyTorch 2.7.1/HIP 6.3.26093, FP32,
704×64 weights and at least 65536 edges. Other shapes retain the original linear
layer. The default is `"torch"`. FP32 reduction order may change; parameter
objects, precision and checkpoint keys retain their original contracts.

The explicit schedules require eager execution and materialized gates.
Example for C64/L2:

```python
configure_model(
    model, training=True, policy="generated", layout="fused_pack",
    message_schedule="roles", species_schedule="indexed",
    graph_reduction="csr", topology_backend="native",
    csr_accumulator_limit=24, csr_slice_search=True,
    geometry_schedule="streamed", message_chunk_edges=32768,
    training_gate_schedule="materialized", mixed_channel_tile=32,
    mixed_geometry_schedule="fused", training_gate_gemm_policy="k100_tiled",
)
```

C128/256, L3 uses `mixed_geometry_schedule="split"` and
`training_adjoint_channel_tile=64`; C256/L3 also sets `csr_accumulator_limit=48`.
K100 native CG/TP energy, forces and mixed second derivatives continue to use the same
CG coefficients and trainable parameters.

### Node/component row layout for message linears

Generated eager training defaults to
`training_message_linear_layout="node_component"`. It packs message-linear
inputs `[N,Cin,2l+1]` into contiguous `[N*(2l+1),Cin]` rows, applies GEMM with
one shared trainable `[Cout,Cin]` weight matrix, then restores the original component
order. This matches the shared GEMM layout of the channel adapters and avoids
strided copies when `einsum` arranges large `[channel,node,angular]` tensors.

```python
configure_model(
    model, training=True, policy="generated", layout="fused_pack",
)
# Explicitly restore the previous layout:
configure_model(model, training=True, training_message_linear_layout="einsum")
```

This option works alongside the TP/gate schedules above. Mixed second derivatives for
weights, inputs and upstream features use ordinary differentiable `linear`,
including force losses and coordinate second backward. Parameter objects,
trainability, state_dict keys and public feature ordering are preserved.
FP32 reduction order may change.

The same default applies to `ForceTrainer` and the training CLI. Use
`--training-message-linear-layout einsum` to select the previous layout.
The layout applies to GPU FP32 without autocast; other dtypes/devices
retain the original calculation. Reconfiguring eager generated training
without this option re-enables the default. Inference, `policy="reference"`
and `auto` reference fallbacks clear the layout flag. Training with
`capture=True` uses the reference route under `auto`; explicit
`policy="generated"` still rejects unsupported training capture.

## Radial checkpointing and per-layer gate schedules

`radial_checkpoint=True` uses non-reentrant activation checkpointing
(`use_reentrant=False`) to recompute radial
MLP activations as needed. It covers the entire original `fc` in the full-gate
route and hidden layers in the streamed-gate route; the final linear projection
retains its gate schedule. Layout-conversion input temporaries are also released
after the message operation, before allocating the returned tensor.
The option defaults off and preserves parameters, precision and model formulas.
See [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint)
for the recomputation mechanism.

`training_gate_schedule` also accepts a tuple/list with one entry per
interaction. A single string still applies to every layer. A speed/memory
tradeoff configuration for two-interaction C128/L3 is:

```python
configure_model(
    model, training=True, policy="generated", layout="fused_pack",
    message_schedule="roles", species_schedule="indexed",
    graph_reduction="csr", topology_backend="device",
    csr_accumulator_limit=24, csr_slice_search=True,
    geometry_schedule="streamed", message_chunk_edges=1_048_576,
    mixed_channel_tile=64,
    training_gate_schedule=("cached", "materialized"),
    radial_checkpoint=True,
)
```

The first layer saves forward gates and recomputes gate gradients as needed;
the second retains the original speed-oriented route. Intermediate activations
of both radial MLPs are recomputed as needed. The memory reduction has an
additional compute cost; measure speed and memory on the same workload.

These options require eager GPU FP32 training with hidden/edge L<=3. Capture
requests raise an error. Compiling/proxy-tensor, functorch, CPU and non-FP32
radial calls retain ordinary computation. Reconfiguring without
`radial_checkpoint`, selecting inference, or selecting `policy="reference"`
clears the option. Parameter objects, trainability, state_dict and checkpoint
keys are preserved. The example is validated for a two-interaction model;
models with different depth need a schedule entry for every interaction.

## Message layouts for energy and force inference

`inference_message_layout="degree"` makes TP produce buffers partitioned by L
that suit subsequent linear layers. First backward for forces reads this layout
directly, avoiding large message rearrangements. This changes storage layout
within native ICTC messages while preserving CG/SC, polynomials, parameters
and precision. Independent per-degree feature slices also become one `split`,
whose backward concatenates the disjoint gradients once.

Example for a model with two interactions and correlation=3, using a larger
gate cache to reduce radial recomputation:

```python
# Caller freezes a dedicated inference model; configure_model preserves trainability.
model.eval().requires_grad_(False)
configure_model(
    model, training=False, policy="generated", layout="fused_pack",
    message_schedule="roles", species_schedule="indexed",
    graph_reduction="csr", topology_backend="device",
    csr_accumulator_limit=24, csr_slice_search=True,
    geometry_schedule="streamed", message_chunk_edges=81_920,
    gate_cache_bytes=5 * 1024**3,
    inference_message_layout="degree",
)
```

`gate_cache_bytes` is a budget per interaction, not a total-model memory cap.
Page size, cache budget and graph size jointly determine the peak. A zero
budget continues to recompute gates by page and is suitable for
capacity-oriented deployment. Speed-oriented results do not imply the same
gain at every node count.

The new layout defaults off. It requires eager GPU FP32, hidden/edge L<=3
and `fused_pack`. Explicit training, capture or high-L requests raise an error.
The streamed-gate route requires frozen radial-network and TP parameters;
the gates still depend on geometry. Unsupported tensor or parameter contexts
retain the original computation. Higher derivatives use
differentiable expressions. Backward does not overwrite the cache in place,
so repeated differentiation remains possible when the graph is retained.

Reconfiguring without this option, switching to `policy="reference"`, or
selecting training clears the new message layout. Parameter objects,
trainability and state_dict keys are preserved.

### Channel tiling for force backward kernels

The configuration above defaults to
`inference_adjoint_channel_tile="auto"`; `64` can also be selected explicitly.
This option reuses existing GPU tiled kernels for first-order backward when
radial-network parameters are frozen and input-feature gradients are needed.
Forward and the first layer's geometry/gate-only backward kernels keep their schedules.
No extra intermediate tensor is added; tiled harmonic-gradient reductions
reuse the existing output buffer. Floating reduction order changes, so
bitwise equality is not guaranteed.

Allowed values are `"auto"` (default), `0` (original schedule), `32` and `64`.
Auto disables tiling during training; explicit 32/64 is invalid with
`training=True`. The option does not freeze parameters or alter differentiable
higher-order fallbacks. Reconfiguring without it restores auto; setting `0`
or selecting reference clears tiling. Both capacity-oriented operation with
`gate_cache_bytes=0` and speed-oriented operation with the original cache
budget can use this option. Validate other devices/configurations separately.

Auto uses bounded structural rules rather than applying a fixed 64-channel
schedule to every model. Current automatic coverage is A100 and RTX 4090 D,
degree layout, streamed geometry and role-specific backward kernels, with at least 4096 edges,
C between 128 and 256, and maximum angular input/output widths both equal to
7 (L3 paths). It selects 64 when the source-level live-value estimate
`ceil(C/32) * max(2*input_width + output_width + 3)` is at least 144; otherwise
it retains the original schedule. This estimate is not a hardware register count.

Both devices use the same structural rule. Standard RTX 4090 and other Ada
cards are not automatically enabled by this policy.

The policy reads only shape/device metadata; it performs no online tuning,
GPU allocation or synchronization. Different atom counts reuse the same
kernels rather than compiling separately for each N. Unknown devices, small
workloads and unvalidated angular widths fall back automatically. Capacity
and speed modes share the same scheduler and differ in gate-cache budget.
This remains an empirical policy with a defined validation range, without
guaranteeing optimal speed for every graph distribution or channel count.

## Nonlinear MACE foundation checkpoints

`mace_ictc.interfaces.foundation_hp.FoundationHP` supports the
`RealAgnosticResidualNonLinearInteractionBlock` architecture used by MH-1.
Node and edge channel counts can differ. Dimensions, paths, cutoff, elements
and heads are read from the checkpoint. Conversion preserves the original
Agnesi radial-distance transform, Ziegler–Biersack–Littmark (ZBL) repulsion,
species-dependent radial networks, layer normalization (LayerNorm), density
normalization, nonlinear gates and all readout heads.

TP and symmetric contraction use ICTC HP. Radial networks are recomputed in
chunks with geometry derivatives retained. HP symmetric contraction
directly consumes e3nn-basis features already produced by the preceding nonlinear
layer, eliminating equivalent round-trip basis transforms and large temporary
layout buffers. `product_basis="bridge"` retains that round trip as a control
route. Here, `product_basis="native"` means the input basis already emitted by
that checkpoint's nonlinear layer, rather than the standalone ICTC Cartesian
feature basis. Both routes use reconstruction checks against the actual checkpoint U
tensors and preserve the original learned coefficients.

Run conversion on the designated remote compute resources:

```bash
python -m mace_ictc.cli.convert_foundation \
  --mace-model mace-mh-1.model --head omat_pbe --out mace-mh-1-ictc-hp.pt
```

The converted file stores an FP64 module. At runtime, explicitly set precision
and freeze weights before enabling HP:

```python
import torch

torch.set_default_dtype(torch.float32)
model = torch.load("mace-mh-1-ictc-hp.pt", map_location="cpu", weights_only=False)
model = model.float().cuda().eval().requires_grad_(False)
plan = model.configure_inference(
    chunk_edges=262144, backward_edges=32768, radial_chunk=32768,
)
model.select_head("omat_pbe")
value = model.forward_vectors(vectors, species, src, dst)
edge_derivative = torch.autograd.grad(value["interaction_energy"].sum(), vectors)[0]
```

`vectors[E,3]` contains displacement vectors from `src` to `dst`, including
periodic image shifts, and must require gradients. `species[N]` contains indices
into the checkpoint element list. `edge_derivative` is an energy derivative
with respect to edge vectors; the caller must assemble atomic forces through
its coordinate/edge mapping.

Results contain `interaction_energy[owned]` and `node_energy[owned]`, where
`node_energy` includes E0 and is accumulated in FP64 even when the model is
cast to FP32. For decomposition by node ownership, `owned` is the integer
count of leading owned nodes, not a mask. It defaults to all input nodes.
Supply `exchange(layer, projected_state)` for distributed hidden features.
For layers after the first, this callback exchanges features after `linear_up`
and returns owned-plus-ghost projected states in the same layout. It must
also propagate reverse communication through autograd.

This interface covers eager force inference with frozen weights. Additional
charge/spin embeddings are outside current coverage. Validate energy and
forces against the actual checkpoint after conversion. Capacity depends on
the model, material density and neighbor count.
