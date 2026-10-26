"""Immutable, source-keyed HIP JIT binaries for concurrent inference workers.

The HIP JIT can relink an existing library in a fresh process, even for the
same source. Never rewrite a successfully published binary while other ranks
may have it mapped. CUDA retains the upstream loader behavior.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch


def _load(loader, name, kwargs, *, file_sources=False):
    if not torch.version.hip:
        return loader(name=name, **kwargs)
    import fcntl
    from torch.utils.cpp_extension import _get_build_directory
    identity = {k: v for k, v in kwargs.items() if k not in ('verbose', 'build_directory')}
    if file_sources:
        identity['source_contents'] = [Path(p).read_text() for p in kwargs['sources']]
    identity.update(torch=torch.__version__, hip=torch.version.hip, cuda=torch.version.cuda,
                    python=list(sys.version_info[:2]), abi=torch._C._GLIBCXX_USE_CXX11_ABI,
                    environment={key: os.environ.get(key) for key in (
                        'CXX', 'CC', 'ROCM_HOME', 'ROCM_PATH', 'HIP_PATH',
                        'HIPCC_COMPILE_FLAGS_APPEND', 'HIPCC_LINK_FLAGS_APPEND',
                        'CPATH', 'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH',
                        'PYTORCH_ROCM_ARCH', 'TORCH_CUDA_ARCH_LIST')})
    suffix = hashlib.blake2b(json.dumps(identity, sort_keys=True).encode(), digest_size=12).hexdigest()
    name = 'hpjit2_' + name + '_' + suffix
    directory = Path(kwargs.get('build_directory') or _get_build_directory(name, False))
    # An explicit build root still needs one directory per immutable identity.
    if kwargs.get('build_directory'):
        directory = directory / name
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / (name + '.so')
    ready = directory / 'published'
    # flock is released on process death, unlike a sentinel lock file. The
    # separate lock also covers load_inline's writes before its internal lock.
    with (directory / 'publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (ready.exists() and binary.exists()):
            module = loader(name=name, **dict(kwargs, build_directory=str(directory)))
            ready.write_text(name + '\n')
            return module
        spec = importlib.util.spec_from_file_location(name, binary)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def load_inline(*, name, **kwargs):
    from torch.utils.cpp_extension import load_inline as torch_load_inline
    return _load(torch_load_inline, name, kwargs)


def load(*, name, **kwargs):
    from torch.utils.cpp_extension import load as torch_load
    return _load(torch_load, name, kwargs, file_sources=True)
