# SPDX-License-Identifier: MIT
"""Run USER-MFFTORCH with the installed native K100 runtime."""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run LAMMPS with native K100 HP execution. Remaining options are passed to LAMMPS.")
    parser.add_argument("-in", "--input", required=True, help="LAMMPS input file")
    parser.add_argument("--plain", action="store_true", help="Disable default Kokkos/HIP graph preparation")
    parser.add_argument("--gpus", type=int, default=None, help="Kokkos GPUs visible per node (default: detected visible count)")
    args, remaining = parser.parse_known_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("HIPBLASLT_ALLOW_TF32", "0")
    # Initialize the same MPI runtime as LAMMPS before Torch loads vendor
    # libraries or starts worker threads. Pass the communicator explicitly.
    from mpi4py import MPI
    import torch
    if not torch.version.hip or not torch.cuda.is_available():
        parser.error("mff-lammps requires HIP-enabled PyTorch on K100")
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    from ictc_o3.backends.k100.fp32_policy import strict_fp32
    strict_fp32()
    from mace_ictc.interfaces.lammps_hp import shutdown
    from lammps import lammps
    flags = []
    if not args.plain:
        count = args.gpus if args.gpus is not None else torch.cuda.device_count()
        if count < 1:
            parser.error("--gpus must be positive")
        flags = ["-k", "on", "g", str(count), "-sf", "kk", "-pk", "kokkos",
                 "newton", "off", "neigh", "full", "comm", "no", "gpu/aware", "off"]
    engine = None
    try:
        engine = lammps(cmdargs=flags + remaining, comm=MPI.COMM_WORLD)
        engine.file(args.input)
    finally:
        shutdown()
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    main(sys.argv[1:])
