"""Command-line interface for data preprocessing."""

import argparse
import os
import sys
import subprocess
import numpy as np
import pandas as pd
from mace_ictc.data.preprocessing import (
    extract_data_blocks,
    fit_baseline_energies,
    compute_correction,
    save_set,
    save_to_h5_parallel,
)


def _random_split_indices(data_size: int, train_ratio: float, seed: int):
    indices = np.arange(data_size)
    np.random.seed(seed)
    train_size = int(train_ratio * data_size)
    val_size = data_size - train_size
    val_indices = np.random.choice(indices, size=val_size, replace=False) if val_size > 0 else np.array([], dtype=int)
    train_mask = ~np.isin(indices, val_indices)
    train_indices = indices[train_mask]
    return train_indices, val_indices


def _source_tail_split_indices(input_file: str, data_size: int, train_ratio: float, seed: int):
    from ase.io import read as ase_read

    atoms_list = ase_read(input_file, index=":")
    if len(atoms_list) != data_size or data_size < 2:
        return None

    source_keys = [atoms.info.get("source") for atoms in atoms_list]
    if not source_keys or any(key is None for key in source_keys):
        return None

    train_size = int(train_ratio * data_size)
    target_val_size = data_size - train_size
    if target_val_size <= 0:
        return np.arange(data_size), np.array([], dtype=int)

    anchor_by_source = {}
    scored_indices = []
    rng = np.random.default_rng(seed)
    for idx, (atoms, source_key) in enumerate(zip(atoms_list, source_keys)):
        source_key = str(source_key)
        if source_key not in anchor_by_source:
            anchor_by_source[source_key] = idx
            continue
        ref_atoms = atoms_list[anchor_by_source[source_key]]
        if (
            len(atoms) != len(ref_atoms)
            or not np.array_equal(atoms.get_atomic_numbers(), ref_atoms.get_atomic_numbers())
        ):
            score = -np.inf
        else:
            diff = atoms.get_positions() - ref_atoms.get_positions()
            score = float(np.sqrt(np.mean(diff * diff)))
        scored_indices.append((score, float(rng.random()), idx))

    if not scored_indices:
        return None

    scored_indices.sort(key=lambda item: (item[0], item[1]), reverse=True)
    target_val_size = min(target_val_size, len(scored_indices))
    val_indices = np.array(sorted(item[2] for item in scored_indices[:target_val_size]), dtype=int)
    indices = np.arange(data_size)
    train_mask = ~np.isin(indices, val_indices)
    train_indices = indices[train_mask]
    return train_indices, val_indices


def _stream_split_xyz(input_file, n, out_paths):
    """Round-robin split an extxyz file into n chunks with bounded memory (one frame at a time).

    Frame c (0-indexed) goes to chunk c % n, so chunk k holds original frames k, k+n, k+2n, ...
    -> chunk k's local frame j maps back to global frame (k + j*n)."""
    fhs = [open(p, "w") for p in out_paths]
    cnt = 0
    try:
        with open(input_file) as f:
            while True:
                head = f.readline()
                if not head:
                    break
                nat = int(head.split()[0])
                comment = f.readline()
                body = [f.readline() for _ in range(nat)]
                o = fhs[cnt % n]
                o.write(head)
                o.write(comment)
                o.writelines(body)
                cnt += 1
    finally:
        for o in fhs:
            o.close()
    return cnt


def _merge_shard_h5(shard_h5_paths, out_h5, max_radius=None):
    """Merge per-shard processed_{prefix}.h5 into out_h5 via HDF5 external links, and build the
    <out_h5>.counts.npz sidecar (node/edge counts indexed by merged sample id) in the same pass.

    External links use ABSOLUTE shard paths -> the shard_*/ dirs must be kept and the dataset
    directory is not relocatable without a re-merge. Returns the total number of merged samples."""
    import h5py
    node_counts = []
    edge_counts = []
    gi = 0
    with h5py.File(out_h5, "w") as fo:
        for src in shard_h5_paths:
            src_abs = os.path.abspath(src)
            with h5py.File(src_abs, "r") as fk:
                n = len(fk.keys())
                for j in range(n):
                    g = fk[f"sample_{j}"]
                    node_counts.append(int(g["pos"].shape[0]))
                    edge_counts.append(int(g["edge_src"].shape[0]))
                    fo[f"sample_{gi}"] = h5py.ExternalLink(src_abs, f"sample_{j}")
                    gi += 1
        # Propagate the neighbor-list cutoff onto the MERGED file so H5Dataset can reject an rcut
        # mismatch (external links do NOT carry the shard files' root attrs). Prefer the explicit
        # arg; else read it back from the first shard (each shard stamped it via save_to_h5_parallel).
        rcut = max_radius
        if rcut is None:
            try:
                with h5py.File(os.path.abspath(shard_h5_paths[0]), "r") as f0:
                    if "max_radius" in f0.attrs:
                        rcut = float(f0.attrs["max_radius"])
            except Exception:
                rcut = None
        if rcut is not None:
            fo.attrs["max_radius"] = float(rcut)
    np.savez(
        out_h5 + ".counts.npz",
        node_counts=np.array(node_counts, dtype=np.int64),
        edge_counts=np.array(edge_counts, dtype=np.int64),
    )
    return gi


def _run_sharded(args):
    """Parallel sharded preprocessing: round-robin split -> N parallel single-shard mff-preprocess
    subprocesses (each loads only its ~1/N, bounded memory) -> merge processed_{train,val}.h5 via
    external links + counts sidecar + merged split-index maps + element-wise-mean global fitted_E0.csv.

    This parallelizes the serial load (extract_data_blocks pulls everything into RAM) and the serial
    single-file H5 write that bottleneck one process on huge datasets. HDF5 cannot be written
    concurrently by multiple processes into one file, so 'shard -> independent writers -> external-link
    merge' is the correct parallel-H5 pattern, not a workaround."""
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # external links on networked FS (Lustre)
    n = int(args.shards)
    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    chunk_dir = os.path.join(out, "_chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_paths = [os.path.join(chunk_dir, f"chunk_{k}.xyz") for k in range(n)]

    # 1) round-robin split (bounded memory)
    if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in chunk_paths):
        print(f"[shards] reusing existing chunk files in {chunk_dir}")
    else:
        print(f"[shards] stream-splitting {args.input_file} into {n} chunks (round-robin)...")
        total = _stream_split_xyz(args.input_file, n, chunk_paths)
        print(f"[shards] split {total} frames into {n} chunks")

    cpu = os.cpu_count() or 1
    if n * max(1, args.num_workers) > cpu:
        print(f"[shards] WARNING: shards({n}) x num-workers({args.num_workers}) = "
              f"{n * args.num_workers} exceeds cpu count ({cpu}); worker processes will be oversubscribed.")

    # 2) launch N single-shard mff-preprocess in parallel (each is a normal --shards 1 run)
    shard_out = [os.path.join(out, f"shard_{k}") for k in range(n)]
    procs = []
    for k in range(n):
        cmd = [
            sys.executable, "-m", "mace_ictc.cli.preprocess",
            "--input-file", chunk_paths[k], "--output-dir", shard_out[k],
            "--shards", "1",
            "--train-ratio", str(args.train_ratio), "--seed", str(args.seed),
            "--max-radius", str(args.max_radius), "--num-workers", str(args.num_workers),
        ]
        if args.atomic_energy_keys:
            cmd += ["--atomic-energy-keys", *[str(x) for x in args.atomic_energy_keys]]
        if args.initial_energy_values:
            cmd += ["--initial-energy-values", *[str(x) for x in args.initial_energy_values]]
        if args.elements:
            cmd += ["--elements", *args.elements]
        if args.max_atom is not None:
            cmd += ["--max-atom", str(args.max_atom)]
        for flag, val in (
            ("--energy-key", args.energy_key), ("--force-key", args.force_key),
            ("--species-key", args.species_key), ("--coord-key", args.coord_key),
            ("--atomic-number-key", args.atomic_number_key),
        ):
            if val:
                cmd += [flag, val]
        logf = open(os.path.join(out, f"shard_{k}.log"), "w")
        print(f"[shards] launch shard {k}: {os.path.basename(chunk_paths[k])} -> {shard_out[k]}")
        procs.append((k, subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT), logf))

    failed = []
    for k, p, logf in procs:
        rc = p.wait()
        logf.close()
        print(f"[shards] shard {k} finished rc={rc}")
        if rc != 0:
            failed.append(k)
    if failed:
        raise RuntimeError(f"[shards] shards {failed} failed; see {out}/shard_*.log")

    # 3) merge processed_{train,val}.h5 via external links + counts sidecar
    for prefix in ("train", "val"):
        srcs = [os.path.join(shard_out[k], f"processed_{prefix}.h5") for k in range(n)]
        srcs = [s for s in srcs if os.path.exists(s)]
        if not srcs:
            print(f"[shards] no processed_{prefix}.h5 across shards; skipping {prefix}")
            continue
        out_h5 = os.path.join(out, f"processed_{prefix}.h5")
        total = _merge_shard_h5(srcs, out_h5, max_radius=args.max_radius)
        print(f"[shards] merged {prefix}: {total} samples -> {out_h5} (+ .counts.npz sidecar)")

    # 4) merged split-index maps (round-robin: global original frame = shard_k + local*N)
    for name in ("train", "val"):
        parts = []
        for k in range(n):
            p = os.path.join(shard_out[k], f"{name}_indices.npy")
            if os.path.exists(p):
                local = np.load(p).astype(np.int64)
                parts.append(k + local * n)
        if parts:
            np.save(os.path.join(out, f"{name}_indices.npy"), np.concatenate(parts))

    # 5) global fitted_E0.csv = element-wise mean of the shard fits (i.i.d. round-robin shards ->
    #    consistent estimate; pass --initial-energy-values or override the csv for exact/rare-element control)
    e0s = []
    for k in range(n):
        p = os.path.join(shard_out[k], "fitted_E0.csv")
        if os.path.exists(p):
            e0s.append(pd.read_csv(p))
    if e0s:
        merged_e0 = e0s[0][["Atom"]].copy()
        merged_e0["E0"] = np.mean(np.stack([df["E0"].values for df in e0s], axis=0), axis=0)
        merged_e0.to_csv(os.path.join(out, "fitted_E0.csv"), index=False)
        print(f"[shards] wrote global fitted_E0.csv (element-wise mean of {len(e0s)} shard fits)")

    # 6) cleanup chunk inputs (big duplicates; not referenced after merge). shard_*/ are KEPT (links point there).
    if not args.keep_chunks:
        for p in chunk_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(chunk_dir)
        except OSError:
            pass

    print(f"[shards] DONE -> {out}/processed_{{train,val}}.h5")
    print(f"[shards] NOTE: the merged dataset uses external links into {out}/shard_*/ (keep those dirs); "
          f"set HDF5_USE_FILE_LOCKING=FALSE when training/reading it.")
    return 0


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description='Preprocess molecular data')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to input XYZ file')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory for preprocessed files (default: data)')
    parser.add_argument('--max-atom', type=int, default=None,
                        help='Legacy padded raw-storage size. Leave unset to use variable-length read_{train,val}.h5.')
    parser.add_argument('--train-ratio', type=float, default=0.95,
                        help='Ratio of training data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--atomic-energy-keys', type=int, nargs='+', default=[1, 6, 7, 8],
                        help='Atomic number keys for energy fitting')
    parser.add_argument('--initial-energy-values', type=float, nargs='+', default=None,
                        help='Initial guess for atomic energies')
    parser.add_argument('--elements', type=str, nargs='+', default=None,
                        help='Element symbols to recognize (default: None, recognizes all elements from periodic table). '
                             'If specified, only these elements will be recognized. Example: --elements C H O N Fe')
    parser.add_argument('--energy-key', type=str, default=None,
                        help='Override the structure-level energy metadata key in extxyz comments (default: energy)')
    parser.add_argument('--force-key', type=str, default=None,
                        help='Override the per-atom vector force property key in extxyz Properties (default search: force/forces/f)')
    parser.add_argument('--species-key', type=str, default=None,
                        help='Override the per-atom species property key in extxyz Properties (default search: species/symbol/element)')
    parser.add_argument('--coord-key', type=str, default=None,
                        help='Override the per-atom coordinate property key in extxyz Properties (default: pos)')
    parser.add_argument('--atomic-number-key', type=str, default=None,
                        help='Override the per-atom atomic-number property key in extxyz Properties (default search: Z/atomic_number)')
    parser.add_argument('--skip-h5', action='store_true',
                        help='Skip neighbor list preprocessing (only save raw data)')
    parser.add_argument('--max-radius', type=float, default=5.0,
                        help='Maximum radius for neighbor search (for H5 preprocessing)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers for H5 preprocessing. With --shards N, this is PER shard '
                             '(total processes ~= N x num-workers; keep that <= cpu count).')
    parser.add_argument('--shards', type=int, default=1,
                        help='Split the input into N shards and preprocess them in PARALLEL: round-robin split -> '
                             'N independent single-shard runs (each loads only ~1/N -> bounded memory) -> merge '
                             'processed_{train,val}.h5 via HDF5 external links + counts sidecar. This parallelizes '
                             'the serial load + single-file write that bottleneck one process on huge datasets. '
                             '1 = normal single process. The merged dataset uses external links into <out>/shard_*/, '
                             'so keep those dirs and set HDF5_USE_FILE_LOCKING=FALSE when training/reading.')
    parser.add_argument('--keep-chunks', action='store_true',
                        help='With --shards, keep the intermediate per-shard input chunk files (default: delete after merge).')

    args = parser.parse_args()

    if getattr(args, "shards", 1) and args.shards > 1:
        return _run_sharded(args)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    print(f"Reading {args.input_file}...")

    # Extract data blocks
    (all_blocks, all_energy, all_raw_energy, all_cells, all_pbcs,
     all_stresses, all_stress_mask) = extract_data_blocks(
        args.input_file,
        elements=args.elements,
        energy_key=args.energy_key,
        force_key=args.force_key,
        species_key=args.species_key,
        coord_key=args.coord_key,
        atomic_number_key=args.atomic_number_key,
        return_stress_mask=True,
    )
    print(f"Total frames: {len(all_blocks)}")

    # Split train/val
    data_size = len(all_blocks)
    split_result = _source_tail_split_indices(args.input_file, data_size, args.train_ratio, args.seed)
    if split_result is None:
        train_indices, val_indices = _random_split_indices(data_size, args.train_ratio, args.seed)
        print("Split mode: random")
    else:
        train_indices, val_indices = split_result
        print("Split mode: source-tail holdout")

    print(f"Split: {len(train_indices)} Train, {len(val_indices)} Val")

    # Save split indices for aligning external labels (e.g. dipole, polarizability)
    # train_indices[i] = original extxyz frame index for processed_train.h5 sample_i
    # val_indices[i] = original extxyz frame index for processed_val.h5 sample_i
    train_indices_path = os.path.join(args.output_dir, 'train_indices.npy')
    val_indices_path = os.path.join(args.output_dir, 'val_indices.npy')
    np.save(train_indices_path, train_indices)
    np.save(val_indices_path, val_indices)
    print(f"Saved {train_indices_path}, {val_indices_path}")

    train_blocks = [all_blocks[i] for i in train_indices]
    train_raw_E = [all_raw_energy[i] for i in train_indices]
    val_blocks = [all_blocks[i] for i in val_indices]
    val_raw_E = [all_raw_energy[i] for i in val_indices]

    # Fit baseline energies
    keys = np.array(args.atomic_energy_keys, dtype=np.int64)
    if args.initial_energy_values is None:
        initial_values = np.array([-0.01] * len(keys), dtype=np.float64)
    else:
        initial_values = np.array(args.initial_energy_values, dtype=np.float64)

    fitted_values = fit_baseline_energies(train_blocks, train_raw_E, keys, initial_values)

    # Save fitted energies
    fitted_e0_path = os.path.join(args.output_dir, 'fitted_E0.csv')
    pd.DataFrame({'Atom': keys, 'E0': fitted_values}).to_csv(fitted_e0_path, index=False)
    print(f"Saved {fitted_e0_path}")

    # Compute corrections
    print("Computing correction energies...")
    train_correction = compute_correction(train_blocks, train_raw_E, keys, fitted_values)
    val_correction = compute_correction(val_blocks, val_raw_E, keys, fitted_values)

    # Save sets
    print("Saving files...")
    save_set('train', train_indices, train_blocks, train_raw_E, train_correction, all_cells, pbc_list=all_pbcs,
             stress_list=all_stresses, stress_mask=all_stress_mask,
             max_atom=args.max_atom, output_dir=args.output_dir)
    save_set('val', val_indices, val_blocks, val_raw_E, val_correction, all_cells, pbc_list=all_pbcs,
             stress_list=all_stresses, stress_mask=all_stress_mask,
             max_atom=args.max_atom, output_dir=args.output_dir)

    print(f"Raw data saved to {args.output_dir}/")

    # Preprocess H5 files (neighbor list computation) - enabled by default
    if not args.skip_h5:
        print("\nComputing neighbor lists (this may take a while)...")
        save_to_h5_parallel('train', args.max_radius, args.num_workers, data_dir=args.output_dir)
        save_to_h5_parallel('val', args.max_radius, args.num_workers, data_dir=args.output_dir)
        print(f"\nDone! All preprocessed files saved in {args.output_dir}/")
        print("You can now run distributed training with:")
        print(f"  torchrun --nproc_per_node=2 -m mace_ictc.cli.train --distributed --data-dir {args.output_dir}")
    else:
        print("\nSkipped neighbor list computation (--skip-h5 was set).")
        print("To complete preprocessing, run:")
        print(f"  mff-preprocess --input-file {args.input_file} --output-dir {args.output_dir}")


if __name__ == '__main__':
    main()
