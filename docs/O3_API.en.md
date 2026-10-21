# ICTC O3 Operator API

English | [Chinese](O3_API.md)

`ictc_o3/` and `mace_ictc/` are sibling packages in the repository. The former
provides standalone operators; the latter provides MACE-compatible models,
training and conversion interfaces. Both are installed with `ictc-o3-hp-k100`
and share the existing HP kernels. Standalone operators can be used without
constructing a MACE model.

```python
from ictc_o3 import (
    EdgeTensorProduct, GraphMessage, PackedGraphMessage,
    direction_harmonics, create_symmetric_contraction,
    configure_symmetric_contraction, channel_linear, normalized_silu,
)
```

These symbols are also available from `ictc_o3.operators`. The original
`mace_ictc.hp` and `mace_ictc.hp.operators` paths export the same classes and
functions, preserving existing calls and serialized operator modules.
Importing `ictc_o3` alone does not load MACE models, initialize a GPU or compile
extensions.

## API overview

Here, `N` is the number of nodes, `E` the number of edges, `C` the channel count,
and `m=2l+1` the number of angular components at degree `l`.

| API | Constructor or factory | Inputs and outputs |
| --- | --- | --- |
| `EdgeTensorProduct` | `.create(channels=..., input_lmax=..., edge_lmax=..., output_lmax=..., layout="ir_mul", device="cuda")`; can also wrap an existing TP with fixed base coefficients | `op(features, harmonics, gates)` → `[E, op.output_width]` |
| `GraphMessage` | `(tp, input_lmax=..., edge_lmax=..., layout="packed", reduction="auto", edge_chunk_size=65536)` | `op(node_blocks, harmonic_blocks, gates, src, dst, mask=None)` → dictionary of node path blocks |
| `PackedGraphMessage` | `.create(channels=..., input_lmax=..., edge_lmax=..., output_lmax=..., reduction="atomic", device="cuda")` | `op(features, harmonics, gates, src, dst, mask=None)` → `[N, op.output_width]` |
| `direction_harmonics` | `(unit_directions, lmax, policy="auto")` | `[E,3]` → list of `[E,2l+1]` blocks, one per degree |
| `create_symmetric_contraction` | `(*, num_elements=..., channels=..., lmax=..., target_lmax=..., correlation=..., training=...)` | Returns a configured, trainable symmetric-contraction module |
| `configure_symmetric_contraction` | `(sc, policy="auto", training=...)` | Selects the execution plan of an existing SC module in place and returns the selection result |
| `channel_linear` | `(features, weight, bias=None)` | `[*,Cin,m]` and `[Cout,Cin]` → `[*,Cout,m]` |
| `normalized_silu` | `(features, scale=1.0)` | Scalar or invariant inputs → `SiLU(features) * scale` |

## Standalone edge tensor product

```python
import torch
from ictc_o3 import EdgeTensorProduct, direction_harmonics

channels, lmax, edges = 32, 2, 257
tp = EdgeTensorProduct.create(
    channels=channels, input_lmax=lmax, edge_lmax=lmax,
    output_lmax=lmax, layout="ir_mul", device="cuda",
)
blocks = {
    l: torch.randn(edges, channels, 2*l+1, device="cuda",
                   dtype=torch.float32, requires_grad=True)
    for l in range(lmax+1)
}
vectors = torch.randn(edges, 3, device="cuda", dtype=torch.float32,
                      requires_grad=True)
harmonics = torch.cat(direction_harmonics(
    torch.nn.functional.normalize(vectors, dim=-1), lmax, policy="generated",
), dim=-1)
gates = torch.randn(edges, tp.gate_width, device="cuda",
                    dtype=torch.float32, requires_grad=True)
output = tp(tp.pack_input(blocks), harmonics, gates)
output_blocks = tp.unpack_output(output)
output.square().mean().backward()
```

The caller selects `channels` and the input, edge and output `lmax` values.
The feature width is `C*(input_lmax+1)**2`; the harmonic width is
`(edge_lmax+1)**2`. Gates accept `[E, tp.gate_width]` or
`[E, len(tp.paths), C]`, in the order given by `tp.paths`.
Outputs preserve each coupling path. A subsequent `channel_linear` can mix
path channels within each degree.

Within each degree/path, `ir_mul` stores `[magnetic,channel]` and `mul_ir`
stores `[channel,magnetic]`. Packing changes the storage layout and retains
the native ICTC angular basis.

## Node graph messages

```python
from ictc_o3 import PackedGraphMessage

message = PackedGraphMessage.create(
    channels=32, input_lmax=2, edge_lmax=2, output_lmax=2,
    reduction="csr", edge_chunk_size=65536, device="cuda",
)
# features: [N, 32*9], ir_mul; harmonics: [E,9]
# gates: [E, message.gate_width]; src/dst: [E], int64 CUDA
node_paths = message(features, harmonics, gates, src, dst)
```

`src` and `dst` use the current node indices and may contain repeated or
unsorted entries. The caller supplies the neighbor list and live radial gates;
the operator performs gather, TP, gating and scatter reduction. The edge TP's
`pack_input` also accepts node blocks with matching channel and degree settings.
For the dictionary interface, use
`GraphMessage(tp, input_lmax=..., edge_lmax=...)`.

## Coverage and gradients

- Standalone edge TP and graph messages use eager CUDA/HIP FP32. Channel counts
  are positive integers and degrees are nonnegative integers. Qualified CSR
  graph forward coverage is L≤3; the existing atomic route supports other degrees.
- The current standalone TP factory selects paths satisfying the triangle
  condition and even `l_in + l_edge + l_out`, corresponding to natural parity
  `p=(-1)^l`. Its interface does not accept arbitrary `(l, parity)` input irreps.
- TP base coefficients are fixed; external gates and inputs remain trainable.
  Wrapping an existing TP preserves its coefficients and does not automatically
  freeze learned parameters. New factories use native ICTC normalization.
- First and mixed second derivatives use standard `torch.autograd.grad` and
  `loss.backward()`. Use `autograd.grad(..., create_graph=True)` to retain the
  graph for geometry derivatives or force-loss backpropagation.
- SC supports native full CG and correlation 1/2/3. Supply `node_type_idx` to use
  the fast factorized route. Direction harmonics require unit directions.
  Channel-linear bias is allowed only for scalar angular blocks. Apply SiLU
  to scalar or invariant features.
- GPU HP execution uses strict FP32 with TF32 disabled. Initial planning and
  JIT compilation are cold-start costs.

Full parameter descriptions, SC input conventions, layouts, higher-degree
coverage and export paths are documented in the
[detailed API guide (Chinese)](HP_API.md).
Import model construction and energy/force helpers from `mace_ictc.hp`, and
the trainer from `mace_ictc.training`.
