"""Single-channel geometry projection shared by tensor products."""
import torch

def bond_first_projection(a: torch.Tensor, b: torch.Tensor, projector: torch.Tensor,
                          angular_backend: str = "torch") -> torch.Tensor:
    """Contract a single-channel geometry leg before broadcasting over channels.

    a: [..., channels, m1], b: [..., 1, m2], projector: [m2, m1*k].
    Unlike broadcasting a batched matmul over channels, this bmm has no repeated
    geometry map and never forms [..., channels, m1*m2].
    """
    magnetic = torch.matmul(b.squeeze(-2), projector).reshape(
        *b.shape[:-2], a.shape[-1], projector.shape[-1] // a.shape[-1]
    )
    from ictc_o3.ops.angular_operator import angular_product
    return angular_product(a, magnetic, angular_backend)

