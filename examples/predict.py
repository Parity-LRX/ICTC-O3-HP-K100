"""Evaluate one ASE-readable structure with a standard or foundation checkpoint.

Run on an allocated K100. Standard HP evaluation expects an FP32 checkpoint;
--reference retains the checkpoint precision and uses the reference operators.
"""
import argparse
import json
from pathlib import Path

import torch
from ase.io import read
from ase.neighborlist import neighbor_list


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", default="prediction.json")
    parser.add_argument("--foundation", action="store_true")
    parser.add_argument("--head", default="omat_pbe")
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    atoms = read(args.structure, index=args.index)
    if len(atoms) == 0 or abs(atoms.cell.volume) < 1e-12:
        raise ValueError("Supply a nonempty structure with a nonsingular cell, including for molecules")
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.foundation:
        model = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        dtype = next(model.parameters()).dtype if args.reference else torch.float32
        model = model.to(device="cuda", dtype=dtype).eval().requires_grad_(False)
        model.select_head(args.head)
        if not args.reference:
            model.configure_inference()
        cutoff = model.max_radius
        known = model.native.atomic_numbers.detach().cpu().tolist()
    else:
        from mace_ictc.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
        from mace_ictc.hp import configure_model
        elements = list(dict.fromkeys(atoms.get_chemical_symbols()))
        loaded = LAMMPS_MLIAP_MFF.from_checkpoint(
            args.checkpoint, element_types=elements, device="cuda",
            operator_plan="checkpoint", torchscript=False,
        )
        model = loaded.wrapper.model.eval().requires_grad_(False)
        dtype = next(model.parameters()).dtype
        if not args.reference and dtype != torch.float32:
            raise ValueError("HP expects FP32: convert with --dtype float32, or use --reference")
        configure_model(model, training=False, policy="reference" if args.reference else "generated",
                        layout="fused_pack")
        cutoff = float(model.max_radius) if hasattr(model, "max_radius") else float(loaded.rcutfac)
        # The standard model returns interaction energies; E0 is stored separately.
        keys = loaded.wrapper.atomic_energy_keys.to(device="cuda", dtype=torch.long)
        offsets = loaded.wrapper.atomic_energy_values.to(device="cuda", dtype=dtype)
        known = keys.detach().cpu().tolist()
    missing = set(atoms.numbers.tolist()) - set(known)
    if missing:
        raise ValueError(f"Atomic numbers absent from checkpoint: {sorted(missing)}")
    src, dst, shifts = neighbor_list("ijS", atoms, cutoff)
    tensor = lambda value, dt: torch.as_tensor(value, device="cuda", dtype=dt)
    positions = tensor(atoms.positions, dtype).detach().requires_grad_(True)
    numbers = tensor(atoms.numbers, torch.long)
    source, target = tensor(src, torch.long), tensor(dst, torch.long)
    cell = tensor(atoms.cell.array, dtype).unsqueeze(0)
    unit_shifts = tensor(shifts, dtype)
    if args.foundation:
        lookup = torch.full((max(known) + 1,), -1, device="cuda", dtype=torch.long)
        lookup[torch.tensor(known, device="cuda")] = torch.arange(len(known), device="cuda")
        species = lookup[numbers]
        vectors = positions[target] - positions[source] + unit_shifts @ cell[0]
        value = model.forward_vectors(vectors, species, source, target)
        energy = value["node_energy"].sum()
        forces = -torch.autograd.grad(value["interaction_energy"].sum() + positions.sum() * 0.0,
                                      positions)[0]
    else:
        from mace_ictc.hp import energy_and_forces
        value = energy_and_forces(model, positions, numbers, torch.zeros_like(numbers),
                                  source, target, unit_shifts, cell)
        lookup = offsets.new_zeros(max(known) + 1)
        lookup[keys] = offsets
        energy = value["energy"].sum() + lookup[numbers].sum()
        forces = value["forces"]
    result = dict(energy_eV=float(energy.detach().cpu()),
                  forces_eV_per_A=forces.detach().cpu().tolist(),
                  atoms=len(atoms), cutoff_A=cutoff,
                  head=args.head if args.foundation else None)
    target_path = Path(args.out)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Energy: {result['energy_eV']:.12g} eV; forces: {len(atoms)} x 3; saved {target_path}")


if __name__ == "__main__":
    main()
