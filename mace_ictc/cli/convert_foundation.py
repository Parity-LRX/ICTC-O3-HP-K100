"""Convert a nonlinear MACE foundation checkpoint for HP force inference.

Run model conversion on the designated compute resource. Example:
  python -m mace_ictc.cli.convert_foundation --mace-model mace-mh-1.model \
    --head omat_pbe --out mace-mh-1-ictc-hp.pt

The output stores an FP64 architecture adapter. Load it, explicitly freeze its
parameters, move to CUDA/HIP FP32, then call configure_inference(). All original
readout heads remain in the checkpoint; --head selects the default evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def convert_file(source, output, *, head="omat_pbe"):
    from mace_ictc.interfaces.foundation_hp import FoundationHP, load_foundation_checkpoint
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        native = load_foundation_checkpoint(source).double().eval()
        converted = FoundationHP(native, head=head).eval()
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(converted, target)
    finally:
        torch.set_default_dtype(previous)
    return dict(source=str(source), output=str(target), head=head, heads=converted.heads,
        cutoff=converted.max_radius, interactions=converted.num_interaction,
        convolution_channels=[layer.channels for layer in converted.convolutions],
        product_channels=[layer.channels for layer in converted.products],
        dtype="float64", numerical_qualification_required=True)


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mace-model", required=True)
    parser.add_argument("--head", default="omat_pbe")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(convert_file(args.mace_model, args.out, head=args.head), indent=2))


if __name__ == "__main__":
    main()
