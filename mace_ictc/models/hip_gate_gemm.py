"""Opt-in FP32 gate GEMMs qualified on K100 / DTK 26.04.

Solution indices are rocBLAS-build-specific, not a portable tuning policy.
Unqualified stacks, shapes, strides, capture and differentiable calls fall back
to PyTorch. Reuse its current-stream BLAS handle without changing its state or
workspace. The qualified solutions request zero additional workspace.
"""
import ctypes as c
from functools import lru_cache

import torch


_SOLUTIONS = {
    ('grad', 16384, 1024): 3321,
    ('grad', 16384, 2944): 3672,
    ('grad', 16384, 5888): 3518,
    ('forward', 32768, 1024): 5745,
    ('forward', 81920, 1024): 4840,
    ('forward', 81920, 5888): 5785,
}


@lru_cache(maxsize=1)
def _runtime():
    if torch.__version__ != '2.7.1' or torch.version.hip != '6.3.26093':
        return None
    try:
        lib = c.CDLL('librocblas.so.4')
        version = c.create_string_buffer(256)
        if lib.rocblas_get_version_string(version, c.c_size_t(len(version))):
            return None
        if version.value != b'4.3.0.ef408db7-dirty':
            return None
    except OSError:
        return None
    from .native_jit import load_inline
    # On the qualified DTK build hipblasCreate directly returns a rocBLAS
    # handle. Keep the ABI bridge behind the exact software-stack guard.
    bridge = load_inline(
        name='ictc_dtk_blas_handle_v1',
        cpp_sources='#include <cstdint>\nuint64_t get_handle();',
        cuda_sources='#include <ATen/cuda/CUDAContext.h>\n#include <cstdint>\n'
                     'uint64_t get_handle(){return reinterpret_cast<uint64_t>('
                     'at::cuda::getCurrentCUDABlasHandle());}',
        functions=['get_handle'], extra_cflags=['-O3'], extra_cuda_cflags=['-O3'])
    return lib, bridge


def gemm(x, weight, out, *, mode):
    """Write a qualified frozen-gate GEMM into out; return False for fallback."""
    if (torch.is_grad_enabled() or not torch.version.hip or not x.is_cuda
            or mode not in ('forward', 'grad')
            or any(t.ndim != 2 or t.dtype != torch.float32 or not t.is_contiguous()
                   or t.device != x.device for t in (x, weight, out))):
        return False
    rows, inner = x.shape
    width = weight.shape[0]
    solution = _SOLUTIONS.get((mode, rows, width))
    if solution is None:
        return False
    cols = width if mode == 'forward' else 64
    if (out.shape != (rows, cols) or weight.shape[1] != 64
            or inner != (64 if mode == 'forward' else width)):
        return False
    if getattr(torch.cuda.get_device_properties(x.device), 'gcnArchName', '').split(':')[0] != 'gfx926':
        return False
    with torch.cuda.device(x.device):
        if torch.cuda.is_current_stream_capturing():
            return False
        runtime = _runtime()
        if runtime is None:
            return False
        lib, bridge = runtime
        handle = c.c_void_p(bridge.get_handle())
        pointer_mode = c.c_int()
        if lib.rocblas_get_pointer_mode(handle, c.byref(pointer_mode)) or pointer_mode.value != 0:
            return False
        alpha, beta = c.c_float(1.), c.c_float(0.)
        status = lib.rocblas_gemm_ex(
            handle, 111 if mode == 'grad' else 112, 111, cols, rows, inner,
            c.byref(alpha), c.c_void_p(weight.data_ptr()), 151, cols if mode == 'grad' else inner,
            c.c_void_p(x.data_ptr()), 151, inner, c.byref(beta), c.c_void_p(out.data_ptr()), 151, cols,
            c.c_void_p(out.data_ptr()), 151, cols, 151, 1, solution, 0)
        if status:
            raise RuntimeError(f'Qualified DTK gate GEMM failed: rocBLAS status {status}')
    return True
