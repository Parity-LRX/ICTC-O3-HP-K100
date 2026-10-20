"""Experimental Cartesian geometry adjoint with per-angular-group reuse."""
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _extension():
    from .native_jit import load
    root = Path(__file__).resolve().parent.parent / "csrc"
    return load(name="mace_ictc_grouped_geometry_v1",
                sources=[str(root / "grouped_edge_geometry.cpp"),
                         str(root / "grouped_edge_geometry.cu")],
                extra_cflags=["-O3"], extra_cuda_cflags=["-O3"], verbose=False)


def grouped_geometry(x, w, g, src, dst, mask, v, p, *, tile_edges=2, warps=4):
    metadata = p.get("grouped_geometry_metadata")
    if metadata is None:
        groups = []; paths = []; outputs = []
        for xb, ni, bb, kt, segments in p["groups"]:
            if ni > 7 or kt > 16 or p["c"] != 64:
                raise ValueError("Grouped geometry currently supports L<=3 and C64")
            groups.append((xb, ni, bb, kt, len(paths)))
            for path, l, start, end in segments:
                base=p["ooffset"][int(l)]+int(p["path_offset"][int(path)])*p["c"]*(2*int(l)+1)
                paths.extend([int(path)]*(int(end)-int(start)))
                outputs.extend(base+k*p["c"] for k in range(int(end)-int(start)))
        metadata = (torch.tensor(groups, dtype=torch.int32, device=x.device).contiguous(),
                    torch.tensor([paths, outputs], dtype=torch.int32, device=x.device).contiguous())
        p["grouped_geometry_metadata"] = metadata
    return _extension().backward(x, w, g, src, dst, mask, v, *metadata, p["bdim"],
                                 bool(p.get("unit_weights", False)), tile_edges, warps)
