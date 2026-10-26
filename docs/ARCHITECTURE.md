# Architecture

The operator library and the MACE framework share one distribution. Model
construction, checkpoint reconstruction and simulation adapters have separate
entry points so applications do not need to import a training command.

## Model construction and checkpoints

- `mace_ictc.models.factory.build_baseline_model` constructs an untrained backbone.
  The public `mace_ictc.hp.create_model` factory delegates to it.
- `mace_ictc.checkpoints.load_checkpoint` reconstructs trained architecture,
  selects saved weights, restores E0 and applies the requested inference plan.
  It returns a `LoadedPotential`, independent of any simulation engine.
- `mace_ictc.checkpoints.adapters` contains shared force and TorchScript wrappers.
- The training CLI parses arguments and invokes the shared factory. Native MFF
  and both core exporters invoke the shared checkpoint loader.
- Historical CLI factory and MLIAP `from_checkpoint` imports remain compatible.
  Existing checkpoint tensor names, normalization and inference freezing remain
  unchanged.

```python
from mace_ictc.checkpoints import load_checkpoint

potential = load_checkpoint(
    "model.pth", ["H", "C", "O"], device="cuda", operator_plan="checkpoint"
)
model = potential.model
```

`operator_plan="checkpoint"` preserves the checkpoint route. Configure the HP
training or inference plan explicitly when changing the intended execution mode.
