"""Optional independent Cartesian channel reduction with per-edge reuse."""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _extension():
    from .native_jit import load
    root=Path(__file__).resolve().parent.parent/"csrc"
    return load(name="mace_ictc_cached_geometry_v1",sources=[str(root/"cached_edge_geometry.cpp"),str(root/"cached_edge_geometry.cu")],
                extra_cflags=["-O3"],extra_cuda_cflags=["-O3"],verbose=False)


def cached_geometry(x,w,g,src,dst,mask,v,p):
    return _extension().backward(x,w,g,src,dst,mask,v,p["bm_channel"],p["component_paths"],bool(p.get("unit_weights",False)))
