"""Optional stride-aware Triton angular contraction and differentiable adjoints.

Design reference: Hamiltonian eager/training angular kernels. No atom-count
dispatch, learned-value cache, atomics, precision reduction or C++ extension.
"""
import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch._subclasses.fake_tensor import is_fake
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:
    @triton.jit
    def _angular(X, B, Y, N, C: tl.constexpr, I: tl.constexpr, K: tl.constexpr,
                 X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr,
                 B0: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr,
                 BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        e, c, k = index // (C*K), (index // K) % C, index % K
        value = tl.full((BLOCK,), 0, tl.float32)
        for i in range(I):
            x = tl.load(X + e*X0 + c*X1 + i*X2, index < N, 0.)
            b = tl.load(B + e*B0 + i*B1 + k*B2, index < N, 0.)
            value = value + x*b
        tl.store(Y + index, value, index < N)

    @triton.jit
    def _map_gradient(X, G, B, E, C: tl.constexpr, I: tl.constexpr, K: tl.constexpr,
                      X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr,
                      G0: tl.constexpr, G1: tl.constexpr, G2: tl.constexpr,
                      BC: tl.constexpr, BO: tl.constexpr):
        output = tl.program_id(0) * BO + tl.arange(0, BO)
        e, i, k = output // (I*K), (output // K) % I, output % K
        c = tl.arange(0, BC)
        mask = (output[None, :] < E*I*K) & (c[:, None] < C)
        x = tl.load(X + e[None, :]*X0 + c[:, None]*X1 + i[None, :]*X2, mask, 0.)
        g = tl.load(G + e[None, :]*G0 + c[:, None]*G1 + k[None, :]*G2, mask, 0.)
        result = tl.sum(x*g, axis=0)
        tl.store(B + output, result, output < E*I*K)


def _forward(x, b):
    out = torch.empty((x.shape[0], x.shape[1], b.shape[2]), device=x.device, dtype=x.dtype)
    if out.numel():
        _angular[(triton.cdiv(out.numel(), 256),)](x, b, out, out.numel(), x.shape[1], x.shape[2], b.shape[2],
            *x.stride(), *b.stride(), 256, num_warps=4, enable_fp_fusion=False)
    return out


def _map_product(x, grad):
    out = torch.empty((x.shape[0], x.shape[2], grad.shape[2]), device=x.device, dtype=x.dtype)
    if out.numel():
        _map_gradient[(triton.cdiv(out.numel(), 16),)](x, grad, out, x.shape[0], x.shape[1], x.shape[2], grad.shape[2],
            *x.stride(), *grad.stride(), triton.next_power_of_2(x.shape[1]), 16,
            num_warps=4, enable_fp_fusion=False)
    return out


class _Angular(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b):
        ctx.save_for_backward(x, b)
        return _forward(x, b)

    @staticmethod
    def backward(ctx, grad):
        x, b = ctx.saved_tensors
        need_x, need_b = ctx.needs_input_grad
        if torch.is_grad_enabled():
            # Force-loss training constructs a graph for coordinate derivatives.
            # This route preserves mixed/second derivatives and learned CG grads.
            dx = torch.matmul(grad, b.transpose(-1, -2)) if need_x else None
            db = torch.matmul(x.transpose(-1, -2), grad) if need_b else None
            return dx, db
        dx = _forward(grad, b.transpose(1, 2)) if need_x else None
        db = _map_product(x, grad) if need_b else None
        return dx, db


class _AngularMap(torch.autograd.Function):
    """X.T @ G with a differentiable analytic adjoint, no stored live cache."""
    @staticmethod
    def forward(ctx, x, grad):
        ctx.save_for_backward(x, grad)
        return _map_product(x, grad)

    @staticmethod
    def backward(ctx, map_grad):
        x, grad = ctx.saved_tensors
        product = _AngularFull.apply if torch.is_grad_enabled() else _forward
        dx = product(grad, map_grad.transpose(-1, -2)) if ctx.needs_input_grad[0] else None
        dg = product(x, map_grad) if ctx.needs_input_grad[1] else None
        return dx, dg


class _AngularFull(torch.autograd.Function):
    """Optional composable Triton adjoints for force-loss double backward.

    The two bilinear maps are each other's analytic adjoints. Nesting their
    autograd nodes keeps mixed derivatives live without tracing native bmm
    copies/reductions in the coordinate backward. No derivatives are detached.
    """
    @staticmethod
    def forward(ctx, x, b):
        ctx.save_for_backward(x, b)
        return _forward(x, b)

    @staticmethod
    def backward(ctx, grad):
        x, b = ctx.saved_tensors
        product = _AngularFull.apply if torch.is_grad_enabled() else _forward
        map_product = _AngularMap.apply if torch.is_grad_enabled() else _map_product
        dx = product(grad, b.transpose(-1, -2)) if ctx.needs_input_grad[0] else None
        db = map_product(x, grad) if ctx.needs_input_grad[1] else None
        return dx, db


def angular_product(x, b, backend="torch"):
    supported = (backend in ("triton", "triton_full") and triton is not None and x.is_cuda and b.is_cuda
                 and x.dtype == b.dtype == torch.float32 and x.ndim == b.ndim == 3
                 and x.device == b.device and x.shape[0] == b.shape[0]
                 and x.shape[-1] == b.shape[-2] and x.shape[1] <= 256
                 and not torch.compiler.is_compiling() and not torch.is_autocast_enabled()
                 and not torch._C._are_functorch_transforms_active()
                 and get_proxy_mode() is None and not is_fake(x) and not is_fake(b))
    if not supported:
        return torch.matmul(x, b)
    function = _AngularFull if backend == "triton_full" else _Angular
    return function.apply(x, b)
