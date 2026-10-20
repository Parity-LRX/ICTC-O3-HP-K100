"""Independent optional shared-memory Cartesian message forward."""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _extension():
    from .native_jit import load
    root=Path(__file__).resolve().parent.parent/"csrc"
    return load(name="mace_ictc_cached_message_v1",sources=[str(root/"cached_edge_message.cpp"),str(root/"cached_edge_message.cu")],
                extra_cflags=["-O3"],extra_cuda_cflags=["-O3"],verbose=False)


def cached_forward(x,sh,w,g,src,dst,mask,out,p,projection):
    _extension().forward(x,sh,w,g,src,dst,mask,projection["forward_indices"],projection["forward_coefficient"],
                         p["km_channel"],out,bool(p.get("unit_weights",False)))
