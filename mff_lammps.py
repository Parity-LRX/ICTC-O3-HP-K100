# SPDX-License-Identifier: MIT
"""Initialize the site's MPI runtime before importing Torch and the HP package."""


def main():
    from mpi4py import MPI  # Initialize MPI before vendor libraries are loaded.
    from mace_ictc.cli.lammps import main as run
    run()


if __name__ == "__main__":
    main()
