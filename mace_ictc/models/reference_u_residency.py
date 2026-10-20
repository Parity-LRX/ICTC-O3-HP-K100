# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Lossless residency for dense reference U after GPU factorization is ready.

Only immutable original/derived U buffers move. Parameters and runtime features
stay on their devices. The dense buffers remain registered with the same names,
shapes, dtypes and values, including for checkpoint serialization.
"""
import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch._subclasses.fake_tensor import is_fake


def constant_key(owner,device,dtype):
    qs=[getattr(owner.basis_bridge,f'q_{l}')
        for l in range(max(owner.lmax,owner.target_lmax)+1)]
    constants=[module.U_tensors(order) for module in owner.symmetric_contractions.contractions
               for order in range(1,module.correlation+1)]
    return tuple((id(value),value._version) for value in [*qs,*constants]),device,dtype


def host_reference_unavailable(model=None):
    """Keep conventional GPU buffers for distributed or replicated modules."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return 'distributed process group initialized'
    candidate=model
    while isinstance(candidate,torch.nn.Module):
        if isinstance(candidate,(torch.nn.DataParallel,torch.nn.parallel.DistributedDataParallel)):
            return 'parallel wrapper requires conventional device buffers'
        child=getattr(candidate,'module',None)
        if child is None:child=getattr(candidate,'model',None)
        if child is candidate:break
        candidate=child
    return None


def _move(owner,device):
    cached=getattr(owner,'_factorized_u_cache',None)
    valid=(cached is not None and cached[1] is not None
           and cached[0]==constant_key(owner,cached[0][1],cached[0][2]))
    changed=False
    for module in owner.symmetric_contractions.contractions:
        for name,value in list(module._buffers.items()):
            if not name.startswith('U_matrix_') or value is None or value.requires_grad:
                continue
            if value.device!=device:
                module._buffers[name]=value.detach().to(device=device)
                changed=True
    # A device copy preserves every coefficient. Other edits/version changes
    # must still invalidate/reconstruct the plan, so refresh only a valid key.
    if changed and valid:
        owner._factorized_u_cache=(constant_key(owner,cached[0][1],cached[0][2]),cached[1])
    return changed


def restore_reference_u(owner,device=None):
    if device is None:device=next(owner.parameters()).device
    return _move(owner,torch.device(device))


def restore_model_reference_u(model):
    from .pure_cartesian_ictd_fix import ICTDBridgeUSymmetricContractionSO3
    for module in model.modules():
        if isinstance(module,ICTDBridgeUSymmetricContractionSO3):
            module._host_reference_u=False
            restore_reference_u(module)


def offload_reference_u(owner,x):
    if not getattr(owner,'_host_reference_u',False) or not x.is_cuda:return
    if host_reference_unavailable() is not None:
        restore_reference_u(owner,x.device)
        return
    _move(owner,torch.device('cpu'))


def prepare_reference_u(owner,x):
    """Restore once for an eager reference call; captures use tensor.to(x)."""
    if (torch.compiler.is_compiling() or get_proxy_mode() is not None
            or torch._C._are_functorch_transforms_active() or is_fake(x)):
        return
    restore_reference_u(owner,x.device)
