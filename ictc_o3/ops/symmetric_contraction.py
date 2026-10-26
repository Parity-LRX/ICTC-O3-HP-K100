"""Trainable symmetric contraction in the fixed Cartesian basis."""
from __future__ import annotations
import torch
from torch import nn
from ictc_o3.basis.bridge import _hidden_irreps, _so3_flat_to_mace_features, SO3ToE3NNBasisBridge
from ictc_o3.ops._mace_symmetric_contraction import MaceSymmetricContraction

class ICTDBridgeUSymmetricContractionSO3(nn.Module):
    """
    Bridge-U symmetric contraction expressed directly in the ICTC basis.

    This is algebraically equivalent to:
      ICTC features -> e3nn basis -> MACE SymmetricContraction -> ICTC basis
    but the per-l basis change is folded into the stored U tensors once at
    initialization. The forward path therefore consumes and returns ICTC-basis
    flat SO3 features. This backend is the stable high-l bridge used when
    pure ICTC U generation is not numerically reliable.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
        use_reduced_cg: bool = False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.use_reduced_cg = bool(use_reduced_cg)
        self.hidden_irreps = _hidden_irreps(self.channels, self.lmax)
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.basis_bridge = SO3ToE3NNBasisBridge(self.channels, max(self.lmax, self.target_lmax))
        self.symmetric_contractions = MaceSymmetricContraction(
            irreps_in=self.hidden_irreps,
            irreps_out=self.target_irreps,
            correlation=int(correlation),
            num_elements=int(num_elements),
            use_reduced_cg=self.use_reduced_cg,
        )
        self._fold_basis_change_into_u_tensors()

    def _input_q(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        blocks = [self.basis_bridge._q(l, dtype=dtype, device=device) for l in range(self.lmax + 1)]
        return torch.block_diag(*blocks)

    def _transform_u_tensor(self, u_tensor: torch.Tensor, output_l: int) -> torch.Tensor:
        dtype = u_tensor.dtype
        device = u_tensor.device
        q_in = self._input_q(dtype=dtype, device=device)
        nu = int(u_tensor.dim() - 1) if int(output_l) == 0 else int(u_tensor.dim() - 2)
        if int(output_l) == 0:
            if nu == 1:
                return torch.einsum("ai,ip->ap", q_in, u_tensor)
            if nu == 2:
                return torch.einsum("ai,bj,ijp->abp", q_in, q_in, u_tensor)
            if nu == 3:
                return torch.einsum("ai,bj,ck,ijkp->abcp", q_in, q_in, q_in, u_tensor)
        q_out = self.basis_bridge._q(output_l, dtype=dtype, device=device)
        if nu == 1:
            return torch.einsum("ro,ai,oip->rap", q_out, q_in, u_tensor)
        if nu == 2:
            return torch.einsum("ro,ai,bj,oijp->rabp", q_out, q_in, q_in, u_tensor)
        if nu == 3:
            return torch.einsum("ro,ai,bj,ck,oijkp->rabcp", q_out, q_in, q_in, q_in, u_tensor)
        # Preserve the established small-order expressions. Higher orders use
        # the same fixed basis map on each tensor leg, without a cubic cap or
        # an einsum alphabet limit. This only lowers immutable U coefficients.
        if int(output_l)==0:
            result=u_tensor
            first_axis=0
        else:
            result=torch.tensordot(q_out,u_tensor,dims=([1],[0]))
            first_axis=1
        for axis in range(first_axis,first_axis+nu):
            result=torch.tensordot(q_in,result,dims=([1],[axis])).movedim(0,axis)
        return result

    def _fold_basis_change_into_u_tensors(self) -> None:
        with torch.no_grad():
            for (mul, ir), contraction in zip(self.target_irreps, self.symmetric_contractions.contractions):
                del mul
                output_l = int(ir.l)
                for nu in range(1, int(contraction.correlation) + 1):
                    name = f"U_matrix_{nu}"
                    old = getattr(contraction, name)
                    new = self._transform_u_tensor(old, output_l)
                    old.copy_(new.to(dtype=old.dtype, device=old.device))
                if getattr(contraction, "_use_scalar_corr3_fast", False) and hasattr(
                    contraction, "refresh_scalar_corr3_fast_buffers"
                ):
                    contraction.refresh_scalar_corr3_fast_buffers()

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        node_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = _so3_flat_to_mace_features(node_feats, self.channels, self.lmax)
        deployment=getattr(self,"_deployment_contraction",None)
        if deployment is not None:
            if node_type_idx is None:node_type_idx=node_attrs.argmax(-1)
            return deployment(x,node_type_idx)
        if node_type_idx is None:
            from ictc_o3.ops.reference_u_residency import prepare_reference_u
            prepare_reference_u(self,x)
            return self.symmetric_contractions(x, node_attrs)
        if getattr(self, "_factorized_contraction", False):
            from ictc_o3.ops.factorized_contraction import factorized_contractions
            factorized = factorized_contractions(self, x, node_type_idx)
            if factorized is not None:
                return factorized
        if getattr(self, "_compact_contraction", False):
            from ictc_o3.ops.compact_contraction import compact_contractions
            compact = compact_contractions(self.symmetric_contractions.contractions, x, node_type_idx)
            if compact is not None:
                return compact
        from ictc_o3.ops.reference_u_residency import prepare_reference_u
        prepare_reference_u(self,x)
        return torch.cat(
            [
                contraction(x, node_attrs, node_type_idx=node_type_idx)
                for contraction in self.symmetric_contractions.contractions
            ],
            dim=-1,
        )

