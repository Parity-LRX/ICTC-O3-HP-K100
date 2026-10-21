"""Native K100 frozen circuits with basis transforms fused into loads/stores.

Features use the original NCI layout; outputs use the original degree blocks.
First feature adjoints share this fusion. Higher derivatives retain the
differentiable polynomial, and trainable coefficient banks use the live route.
"""
import torch


def small_force_workload(x,channels,*,requires_grad=None):
    if (not torch.version.hip or not x.is_cuda or not torch.is_grad_enabled()
            or not (x.requires_grad if requires_grad is None else requires_grad)
            or not x.shape[0]):
        return False
    properties=torch.cuda.get_device_properties(x.device)
    return (getattr(properties,'gcnArchName','').split(':')[0]=='gfx926'
            and x.shape[0]*channels<properties.multi_processor_count*2048)


class _Fused(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,bank,index,plan):
        from .hip_circuit import raw_basis
        ctx.save_for_backward(x,bank,index);ctx.plan=plan
        return raw_basis(x,bank,index,plan)

    @staticmethod
    def backward(ctx,grad):
        from .hip_circuit import raw_basis
        x,bank,index=ctx.saved_tensors;plan=ctx.plan
        if torch.is_grad_enabled():
            from .species_circuit import reference
            local=x.clone();output=reference(local@plan['q'],bank,index,plan)
            parts=output.split(plan['sizes'],-1)
            output=torch.cat([(part@matrix.T).reshape(x.shape[0],x.shape[1]*part.shape[-1])
                              for part,matrix in zip(parts,plan['qout'])],-1)
            dx=torch.autograd.grad(output,local,grad,create_graph=True)[0]
        else:
            dx=raw_basis(x,bank,index,plan,1,grad.contiguous())
        return dx,None,None,None


def fused(x,bank,index,plan):
    if bank.requires_grad or plan['q'].requires_grad or any(q.requires_grad for q in plan['qout']):
        return None
    if index.shape!=(x.shape[0],) or index.device!=x.device or index.dtype!=torch.int64:
        raise ValueError('Species indices must be int64 [N] on the feature device')
    if small_force_workload(x,x.shape[1]):
        # Whole-model first-adjoint timings qualify packing for small node /
        # channel workloads. Energy-only calls retain the direct fused route.
        from .hip_packed_basis import fused as packed
        return packed(x,bank,index,plan)
    torch._assert_async(((index>=0)&(index<bank.shape[0])).all(),'Species index outside the coefficient bank')
    return _Fused.apply(x.contiguous(),bank.contiguous(),index.contiguous(),plan)
