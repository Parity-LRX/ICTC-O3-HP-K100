"""Data handling: H5 dataset, collate, make_fx size-bucketing, preprocessing.

The training data stack ported from FSCETP so MACE-ICTC can train (with make_fx
size-bucketing), not only deploy. The dataset itself does NO normalization:
energies are stored already-baseline-subtracted by preprocessing and the per-type
E0 is re-added in the trainer; ``restore_force``/``restore_energy`` are identity.
"""

from mace_ictc.data.datasets import (
    CustomDataset,
    H5Dataset,
    OnTheFlyDataset,
    compute_graph_worker,
)
from mace_ictc.data.collate import collate_fn_h5, my_collate_fn, on_the_fly_collate
from mace_ictc.data.bucket_sampler import BucketBatchSampler

__all__ = [
    "CustomDataset",
    "H5Dataset",
    "OnTheFlyDataset",
    "compute_graph_worker",
    "collate_fn_h5",
    "my_collate_fn",
    "on_the_fly_collate",
    "BucketBatchSampler",
]
