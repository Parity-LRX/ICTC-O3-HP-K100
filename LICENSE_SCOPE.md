# License scope

Project-owned code and documentation in ICTC-O3-HP-K100 are licensed under the
[MIT License](LICENSE), unless a file carries a different license notice.
This includes the `ictc_o3/` operator APIs, the MACE-ICTC framework, HP kernels,
training code, and project-owned deployment and LAMMPS integration code.
Commercial use is permitted under the MIT terms.

## Third-party code and dependencies

Original third-party copyright notices and license terms are retained:

- Earlier MACE-ICTC code: [MIT](LICENSES/MACE-ICTC-MIT.txt).
- MACE-derived Clebsch-Gordan and symmetric-contraction code, including
  `mace_ictc/models/_mace_cg.py` and `_mace_symmetric_contraction.py`:
  [MIT](LICENSES/MACE-MIT.txt), with the original author notices retained.
- e3nn: [MIT](LICENSES/e3nn-MIT.txt).
- `mace_ictc/models/_cueq_cg_tools.py`: copyright (c) 2024 NVIDIA CORPORATION
  & AFFILIATES, [Apache License 2.0](LICENSES/Apache-2.0.txt).

Dependencies retain their upstream licenses. LAMMPS itself remains under GPLv2;
its license applies to LAMMPS and distributions that include it. See the
[official LAMMPS license guidance](https://docs.lammps.org/Intro_opensource.html).

## Pretrained models and data

Pretrained model weights and training datasets retain their upstream licenses.
The MIT license for this software does not change those terms.
