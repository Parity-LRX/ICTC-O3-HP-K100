"""Recompute radial MLP activations without changing modules or parameters."""
import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch._subclasses.fake_tensor import is_fake


def radial_apply(fc,value,*,hidden=False,checkpoint=False,gemm_policy='torch'):
    """Evaluate the original sequence, optionally excluding its last linear."""
    def evaluate(x):
        if (not hidden and gemm_policy=='k100_tiled' and isinstance(fc,torch.nn.Sequential)
                and len(fc) and type(fc[-1]) is torch.nn.Linear
                and fc[-1].in_features==64 and fc[-1].out_features==704
                and x.ndim==2 and x.shape[0]>=65536
                and x.is_cuda and x.dtype==torch.float32
                and not torch.compiler.is_compiling() and get_proxy_mode() is None
                and not is_fake(x) and not torch._C._are_functorch_transforms_active()
                and not torch.is_autocast_enabled(x.device.type)):
            from .training_linear import gate_linear
            for index in range(len(fc)-1):x=fc[index](x)
            return gate_linear(x,fc[-1].weight,fc[-1].bias,policy=gemm_policy)
        if not hidden:return fc(x)
        for index in range(len(fc)-1):x=fc[index](x)
        return x

    if (not checkpoint or not torch.is_grad_enabled() or not value.is_cuda
            or value.dtype!=torch.float32 or torch.compiler.is_compiling()
            or torch.is_autocast_enabled(value.device.type)
            or get_proxy_mode() is not None or is_fake(value)
            or torch._C._are_functorch_transforms_active()):
        return evaluate(value)
    from torch.utils.checkpoint import checkpoint as recompute
    return recompute(evaluate,value,use_reentrant=False)
