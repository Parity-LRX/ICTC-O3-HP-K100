# Fine-tuning MACE-MH-1 with ICTC on K100

Use `FoundationTrainHP` to train the original nonlinear MACE-MH-1 architecture
through ICTC operators. All canonical learned parameters stay trainable and
all original readout heads are retained; `head` selects the prediction head.
The frozen-weight `FoundationHP` converter and `convert_foundation` CLI are
for inference and should not be used to prepare a trainable model.

This guide uses the Python API. The generic `mff-train` CLI builds the library's
own backbone; it does not automatically wrap an MH-1 foundation checkpoint.

## Environment and input files

Install from the repository root in an environment with the matching K100
PyTorch, HIP/DTK and Triton runtime:

```bash
python -m pip install -e '.[hp,train,conversion]'
```

Prepare an official `mace-mh-1.model`, ASE-readable labelled structures such as
`train.extxyz`, and `E0s.json`. Energies must be in eV, positions in Å and forces
in eV/Å. `E0s.json` maps atomic-number strings to isolated-atom energies in eV.
Use references consistent with the DFT settings of the labelled dataset.

The example reads labels with `atoms.get_potential_energy()` and
`atoms.get_forces()`. If labels use custom names such as `REF_energy` and
`REF_forces`, adapt these two accesses. An optional Boolean per-atom
`force_mask` includes or excludes atoms from the force loss. By default all
atoms, including Na, contribute; no element is silently excluded.

Run data preparation, compilation and training on allocated compute resources.
On a Slurm cluster, use `sbatch` and the configured GPU partition.

## Load and configure the trainable model

```python
from dataclasses import asdict
import json
import math
import random

import numpy as np
import torch
from ase.io import read
from mace import data as mace_data
from mace.calculators import MACECalculator
from mace.tools.torch_geometric import Batch

from mace_ictc.hp import check_model, configure_model
from mace_ictc.interfaces.foundation_hp import load_foundation_checkpoint
from mace_ictc.interfaces.foundation_training import (
    FoundationTrainHP,
    FoundationTrainingConfig,
)

torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(73)

head = "omat_pbe"  # Must be present in the checkpoint's heads.
with open("E0s.json", encoding="utf-8") as handle:
    e0 = {int(z): float(value) for z, value in json.load(handle).items()}

native = load_foundation_checkpoint("mace-mh-1.model")
# Retain MACE's element table, cutoff and head metadata for graph construction.
calculator = MACECalculator(
    models=[native], head=head, device="cuda", default_dtype="float32",
    enable_cueq=False, compute_stress=False,
)
native = calculator.models[0]
head_index = list(native.heads).index(head)
with torch.no_grad():
    table = native.atomic_energies_fn.atomic_energies
    for index, z in enumerate(native.atomic_numbers.tolist()):
        if int(z) in e0:
            table[head_index, index] = e0[int(z)]

config = FoundationTrainingConfig.k100_mh1(mode="speed")
# Use mode="memory" to recompute gates instead of retaining materialized gates.
model = FoundationTrainHP(native, head=head, config=config).float().cuda()
support = check_model(model, training=True)
if not support["supported"]:
    raise RuntimeError(support["reasons"])
plan = configure_model(
    model, training=True, policy="generated", foundation_config=config,
)
parameters = list(model.native.parameters())
optimizer = torch.optim.Adam(parameters, lr=1e-4, amsgrad=True)
```

Create the adapter and select the device/dtype before creating the optimizer.
The adapter makes a copy of the supplied native model; optimize
`model.native.parameters()`, rather than the original calculator's model.
Reconfiguration preserves the adapter's canonical parameter objects and keys.

The eager training interface requires a GPU FP32 network with hidden/edge
angular degree at most 3. Energy terms and reductions use FP64 in both K100
modes. This guide covers energy and force parameter training; stress and
external position Hessians are outside this adapter's interface.

## Build a batch and form the energy/force loss

```python
keys = mace_data.KeySpecification(
    info_keys=calculator.info_keys, arrays_keys=calculator.arrays_keys,
)


def make_batch(frames):
    graphs = [mace_data.AtomicData.from_config(
        mace_data.config_from_atoms(atoms, key_specification=keys, head_name=head),
        z_table=calculator.z_table,
        cutoff=calculator.r_max,
        heads=calculator.available_heads,
    ) for atoms in frames]
    graph = Batch.from_data_list(graphs).to("cuda")
    # Keep the exact reference values outside the FP32 model buffers.
    e0_sum = [math.fsum(e0[int(z)] for z in atoms.numbers) for atoms in frames]
    target_interaction = torch.tensor(
        [float(atoms.get_potential_energy()) - reference
         for atoms, reference in zip(frames, e0_sum)],
        device="cuda", dtype=torch.float64,
    )
    target_forces = torch.as_tensor(
        np.concatenate([atoms.get_forces() for atoms in frames]),
        device="cuda", dtype=torch.float32,
    )
    mask = torch.as_tensor(np.concatenate([
        np.asarray(atoms.arrays.get("force_mask", np.ones(len(atoms))), dtype=bool)
        for atoms in frames
    ]), device="cuda", dtype=torch.bool)
    if not bool(mask.any()):
        raise ValueError("This batch has no force labels selected by force_mask")
    return graph, target_interaction, target_forces, mask


def train_step(frames):
    graph, target_energy, target_forces, mask = make_batch(frames)
    counts = (graph.ptr[1:] - graph.ptr[:-1]).to(torch.float64)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    prediction = model(
        graph.to_dict(), training=True, compute_force=True,
        compute_stress=False, compute_virials=False,
    )
    assert prediction["interaction_energy"].dtype == torch.float64
    energy_loss = (
        (prediction["interaction_energy"] - target_energy) / counts
    ).square().mean()
    force_loss = (
        prediction["forces"][mask] - target_forces[mask]
    ).square().mean()
    loss = 10.0 * energy_loss + 10.0 * force_loss
    # training=True retains the geometry derivative graph needed by force loss.
    loss.backward(inputs=parameters)
    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


# Illustrative in-memory loader. Use streaming/sharded data for large datasets.
training_frames = read("train.extxyz", index=":")
batch_size = 2
order = random.Random(73)
for epoch in range(20):
    order.shuffle(training_frames)
    for start in range(0, len(training_frames), batch_size):
        train_step(training_frames[start:start + batch_size])
```

The example weights and learning rate are starting choices, not fitted
hyperparameters for another dataset. Keep validation/test structures separate
and use validation metrics to choose a checkpoint. Fit any initial per-atom
energy shift using training frames only. Preserve trajectory/group boundaries
when splitting correlated MD samples.

`interaction_energy` already includes the checkpoint's scale/shift and pair
term, but excludes isolated-atom E0. Train it against `E_DFT - sum(E0)` as above.
For total energies, add the same external E0 sum in FP64. Do not add E0 twice
or cast a completed FP32 sum to FP64: lost precision cannot be recovered.
Features, readout GEMMs, forces and learned parameters stay FP32.

## Validation, checkpointing and native MACE weights

For validation, use `model.eval()` and `training=False` in the forward call.
Keep autograd enabled when requesting forces; do not wrap force prediction in
`torch.no_grad()`. Set `training=True` when comparing force-loss parameter
gradients against native MACE.

```python
torch.save({
    "native_state_dict": {
        name: value.detach().cpu()
        for name, value in model.native.state_dict().items()
    },
    "head": head,
    "foundation_config": asdict(config),
    "E0s_eV": e0,
    "optimizer_state_dict": optimizer.state_dict(),
}, "finetuned.pt")
```

Restore the canonical state into the same original foundation architecture,
then rebuild the adapter before constructing/loading its optimizer:

```python
state = torch.load("finetuned.pt", map_location="cpu", weights_only=False)
native = load_foundation_checkpoint("mace-mh-1.model")
native.load_state_dict(state["native_state_dict"], strict=True)
model = FoundationTrainHP(
    native, head=state["head"],
    config=FoundationTrainingConfig(**state["foundation_config"]),
).float().cuda()
```

Export `model.native`, rather than the ICTC wrapper, when another tool expects
a serialized native MACE model. Carry the selected head and external E0 table
with the export. The native FP32 atomic-energy buffer may round E0 values;
exact total-energy evaluation must reconstruct the external sum in FP64.

For a native comparison with matching energy reductions, use
`mace_ictc.interfaces.foundation_energy.foundation_forward_fp64`. It promotes
each atomic energy term before layer/graph summation. Use identical weights,
heads, geometry, E0 convention and loss masks when checking energy, forces and
parameter gradients before a longer run. The exported native model does not
retain the helper's temporary hooks.

## Submit on K100

Save the example in a training script and run it inside an allocated job.
Select the account, partition, GPU resource name and vendor environment for
the target cluster. For a cluster exposing K100 as `dcu`:

```bash
#!/bin/bash -l
#SBATCH --partition=DCU_K100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=finetune-%j.log
set -euo pipefail
# Activate the compatible PyTorch/HIP/Triton environment here.
export OMP_NUM_THREADS=4
export HIPBLASLT_ALLOW_TF32=0
python -u finetune_mh1.py
```

Submit with `sbatch finetune_mh1.sbatch`. First use a short allocated job to
check your dataset, memory use and native/ICTC agreement. Benchmark outputs,
logs and dataset-specific reports belong in local task storage.
