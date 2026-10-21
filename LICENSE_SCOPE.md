# License scope — MACE-ICTC-HP 0.2.4

This release offers the newly added, project-owned HP implementations under
the unmodified PolyForm Noncommercial License 1.0.0 in [LICENSE](LICENSE).
Rights holders can separately grant commercial permissions according to
[COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md). That document is a procedure,
not an automatic commercial license. The standard PolyForm exceptions remain.

## New HP implementations

The PolyForm offer covers the following new HP components and their newly
added integration changes, to the extent their copyright is controlled by
the project rights holders:

- `ictc_o3/`: documented standalone operator APIs and their HP implementations.
- `mace_ictc/hp/`: whole-model APIs and compatibility imports for the operators.
- `mace_ictc/models/generated_cartesian_message.py`,
  `generated_message_second.py`, `generated_direction_harmonics.py`:
  independent CUDA/Triton source generation and adjoints.
- `generated_csr_forward.py`, `species_linear.py`, `fp32_policy.py` in
  `mace_ictc/models/`: destination-owned FP32 reduction, shared species GEMMs
  and the strict FP32 execution policy.
- `reference_u_residency.py`: lossless reference-constant residency and
  cache/version-preserving device restoration.
- `angular_operator.py`, `bond_projection.py`, `banked_contraction.py`,
  `compact_contraction.py`, `factorized_contraction.py`, `factorized_circuit.py`,
  `partitioned_circuit.py`, `staged_circuit.py`, `bounded_quadratic.py`,
  `polynomial_gemm.py`, `monomial_operator.py`, `fused_polynomial_matmul.py`,
  `fused_message.py`, `rematerialized_message.py`, `streamed_gate_message.py`,
  `fused_path_message.py`, `fused_activation.py`, `direct_harmonic_gradient.py`,
  `csr_group_message.py`, `message_layout.py`, `adjoint_requirements.py`,
  `operator_plans.py` in `mace_ictc/models/`: new planning, layout, contraction
  and graph-operator implementations.
- `mace_ictc/interfaces/deployment_triton.py`, `deployment_sparse.py`: prepared
  HP deployment operators and bounded high-degree contractions/adjoints.
- `mace_ictc/csrc/cached_edge_*`, `grouped_edge_geometry.*`, `csr_message.*`:
  new HP C++/CUDA kernels.
- Newly added HP selection/layout integration portions of the existing model,
  training, checkpoint and export files, plus newly created HP usage examples.

The module/path layout itself does not remove pre-existing third-party rights.
Where a listed component includes pre-existing permissive material, its original
copyright and permission notices continue to apply to that material.

## Preserved MIT rights

The earlier MACE-ICTC implementation remains covered by its existing MIT offer,
reproduced in [LICENSES/MACE-ICTC-MIT.txt](LICENSES/MACE-ICTC-MIT.txt). Existing
MIT portions in modified files retain those permissions. This release does
not retroactively revoke earlier recipients' MIT rights or claim that merely
renaming the package makes previously MIT-licensed code noncommercial.

In particular, the original `ictd_irreps.py`, `pure_cartesian_ictd_fix.py`,
`csrc/ictd_tp.cpp` and `csrc/ictd_tp_cuda.cu` baseline material keeps its MIT
permissions; new HP additions are distinguished by the scope above.

`models/_mace_cg.py` and `models/_mace_symmetric_contraction.py` retain their
MACE author notices and MIT permission. [LICENSES/MACE-MIT.txt](LICENSES/MACE-MIT.txt)
contains the upstream MACE notice. The e3nn notice is in
[LICENSES/e3nn-MIT.txt](LICENSES/e3nn-MIT.txt); use of e3nn as a dependency does
not replace that dependency's license with PolyForm. All third-party source
notices, patents and model-weight licenses retain their original scope.

Package metadata uses `PolyForm-Noncommercial-1.0.0 AND MIT` to describe the
standard noncommercial distribution route for this combined Python edition.
It does not mean that every source line is relicensed under both terms, or
that MIT alone licenses the newly restricted HP components. An alternative
commercial grant can cover only rights that its issuer is entitled to grant.

## LAMMPS integration

The separate `lammps_user_mfftorch` integration is not automatically relicensed
as PolyForm by this Python release. Existing rights and third-party notices
are retained. LAMMPS itself is GPLv2; interface code, linking and distribution
of a combined executable or required runtime libraries need their own license
compatibility assessment. This package does not assert that a PolyForm grant
or a commercial Python grant automatically permits any GPL combination.
See the [official LAMMPS license guidance](https://docs.lammps.org/Intro_opensource.html).

This release provides source and license texts, not a new license for third-party
training data or pretrained potentials. Rights holders should confirm institutional
and contributor authority before publishing or signing commercial agreements.
