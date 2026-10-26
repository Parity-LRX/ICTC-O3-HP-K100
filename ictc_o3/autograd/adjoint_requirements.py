"""Respect the current autograd target graph, with conservative fallback.

ctx.needs_input_grad reflects requires_grad at forward construction. A grad
call targeting coordinates may prune parameter-only ancestors (and conversely).
The engine query also treats explicitly requested intermediate tensors as
needed. Unsupported tracing/runtime contexts retain every declared gradient.
"""
import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode


def required_gradients(inputs,declared):
    if (torch.compiler.is_compiling() or get_proxy_mode() is not None
            or torch._C._are_functorch_transforms_active()
            or not hasattr(torch._C,"_will_engine_execute_node")):
        return tuple(declared)
    result=[]
    for tensor,needed in zip(inputs,declared):
        if not needed:
            result.append(False);continue
        try:
            node=torch.autograd.graph.get_gradient_edge(tensor).node
            result.append(bool(torch._C._will_engine_execute_node(node)))
        except (RuntimeError,AttributeError,AssertionError):
            result.append(True)
    return tuple(result)
