# ICTC-O3-HP-K100

K100 (`gfx926`) HIP optimized Cartesian tensor operators, atomistic force
training and inference. Version **0.1.0**.

Distribution: `ictc-o3-hp-k100`. Python namespace: `mace_ictc`, retained for
model and checkpoint compatibility.

## Installation

Use an isolated environment with a compatible K100 vendor PyTorch, HIP/DTK
and Triton stack, then install:

```bash
git clone https://github.com/Parity-LRX/ICTC-O3-HP-K100.git
cd ICTC-O3-HP-K100
python -m pip install -e '.[hp,train,conversion]'
```

The `hp` extra supplies Ninja and preserves the installed vendor Triton.
GPU execution and JIT compilation need the matching HIP compiler and runtime.
On Slurm clusters, run computation through `sbatch` on an allocated GPU.

## Features

- Tensor products, graph aggregation, direction harmonics and symmetric contractions.
- Native HIP first and mixed derivatives for energy-and-force training.
- Streamed inference gates, bounded adjoints and grouped element linears.
- Nonlinear MACE-MH-1 conversion with all original readout heads retained.
- LAMMPS interfaces and model export utilities.

```python
from mace_ictc.hp import create_model, configure_model

model = create_model(channels=64, lmax=2, num_interactions=2, correlation=3)
configure_model(
    model, training=True, policy="generated", layout="fused_pack",
    training_message_linear_layout="node_component",
)
```

See the [API guide](docs/HP_API.md) for training, inference and conversion.
The current MH-1 foundation adapter supports frozen-weight force inference;
full MH-1 fine-tuning requires a training adapter.

## License

Derived from [ICTC-O3-HP](https://github.com/Parity-LRX/ICTC-O3-HP).
Existing PolyForm Noncommercial 1.0.0 and MIT license scopes are retained;
see [LICENSE_SCOPE.md](LICENSE_SCOPE.md) and [NOTICE](NOTICE).
Pretrained model weights have their own upstream licenses.
