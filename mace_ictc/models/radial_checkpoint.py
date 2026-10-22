"""Recompute radial MLP activations without changing modules or parameters."""
import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch._subclasses.fake_tensor import is_fake


class _FrozenChunks(torch.autograd.Function):
    """Bound row-independent frozen MLP temporaries in both directions."""
    @staticmethod
    def forward(ctx, value, evaluate, rows):
        ctx.evaluate, ctx.rows = evaluate, rows
        ctx.save_for_backward(value)
        if not value.shape[0]:
            return evaluate(value)
        first = evaluate(value[:rows])
        result = first.new_empty((value.shape[0], *first.shape[1:]))
        result[:rows].copy_(first)
        del first
        for begin in range(rows, value.shape[0], rows):
            result[begin:begin+rows].copy_(evaluate(value[begin:begin+rows]))
        return result

    @staticmethod
    def backward(ctx, gradient):
        value, = ctx.saved_tensors
        higher = torch.is_grad_enabled()
        result = [] if higher else torch.empty_like(value)
        for begin in range(0, value.shape[0], ctx.rows):
            end = min(begin + ctx.rows, value.shape[0])
            with torch.enable_grad():
                part = value[begin:end]
                if not higher:
                    part = part.detach().requires_grad_(True)
                output = ctx.evaluate(part)
                derivative, = torch.autograd.grad(output, part, gradient[begin:end],
                    create_graph=higher)
            if higher:
                result.append(derivative)
            else:
                result[begin:end].copy_(derivative)
            del output, derivative, part
        if higher:
            result = torch.cat(result) if result else value * 0
        return result, None, None


def radial_apply(fc,value,*,hidden=False,checkpoint=False,gemm_policy='torch',chunk_rows=0):
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
    if (chunk_rows > 0 and value.ndim == 2 and isinstance(fc,torch.nn.Sequential)
            and all(not parameter.requires_grad for parameter in fc.parameters())):
        # Native capacity inference supplies a row-independent radial sequence.
        # The full output remains necessary for the streamed message operator;
        # only the prefix's temporary activations and its VJP are chunked.
        return _FrozenChunks.apply(value,evaluate,int(chunk_rows))
    from torch.utils.checkpoint import checkpoint as recompute
    return recompute(evaluate,value,use_reentrant=False)
