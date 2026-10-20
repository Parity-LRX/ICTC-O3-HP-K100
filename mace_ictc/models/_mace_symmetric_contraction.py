###########################################################################################
# Implementation of the symmetric contraction algorithm presented in the MACE paper
# (Batatia et al., MACE: Higher Order Equivariant Message Passing Neural Networks
# for Fast and Accurate Force Fields, Eq. 10 and 11)
# Authors: Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
#
# Local minimal copy for spherical-fix native MACE contraction.
###########################################################################################

from __future__ import annotations

import os
from typing import Dict, Optional, Union

import opt_einsum_fx
import torch
import torch.fx
from e3nn import o3
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode

from mace_ictc.models._mace_cg import U_matrix_real

BATCH_EXAMPLE = 10
ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]
_USE_SCALAR_CORR3_FAST = os.environ.get("ICTD_USE_SCALAR_CORR3_CONTRACTION", "1") == "1"


@compile_mode("script")
class MaceSymmetricContraction(CodeGenMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: Union[int, Dict[str, int]],
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        use_reduced_cg: bool = False,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        num_elements: Optional[int] = None,
    ) -> None:
        super().__init__()
        del path_normalization
        if irrep_normalization is None:
            irrep_normalization = "component"
        assert irrep_normalization in ["component", "norm", "none"]

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        if num_elements is None:
            raise ValueError("num_elements must be provided for MACE-style contraction")

        if not isinstance(correlation, dict):
            corr = int(correlation)
            correlation = {irrep_out: corr for irrep_out in self.irreps_out}

        if internal_weights is None:
            internal_weights = True
        if shared_weights is None:
            shared_weights = True
        assert shared_weights or not internal_weights

        self.internal_weights = bool(internal_weights)
        self.shared_weights = bool(shared_weights)
        self.contractions = torch.nn.ModuleList()
        for irrep_out in self.irreps_out:
            self.contractions.append(
                _Contraction(
                    irreps_in=self.irreps_in,
                    irrep_out=o3.Irreps(str(irrep_out.ir)),
                    correlation=int(correlation[irrep_out]),
                    internal_weights=self.internal_weights,
                    num_elements=int(num_elements),
                    weights=self.shared_weights,
                    use_reduced_cg=use_reduced_cg,
                )
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        outs = [contraction(x, y) for contraction in self.contractions]
        return torch.cat(outs, dim=-1)


@compile_mode("script")
class _Contraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irrep_out: o3.Irreps,
        correlation: int,
        internal_weights: bool = True,
        use_reduced_cg: bool = False,
        num_elements: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if num_elements is None:
            raise ValueError("num_elements must be provided")

        self.num_features = irreps_in.count((0, 1))
        self.coupling_irreps = o3.Irreps([irrep.ir for irrep in irreps_in])
        self.correlation = int(correlation)
        self.output_lmax = int(irrep_out.lmax)
        self.internal_weights = bool(internal_weights)
        self.shared_weights = bool(weights) if isinstance(weights, bool) else weights is not None
        dtype = torch.get_default_dtype()

        path_weight = []
        for nu in range(1, self.correlation + 1):
            u_matrix = U_matrix_real(
                irreps_in=self.coupling_irreps,
                irreps_out=irrep_out,
                correlation=nu,
                use_cueq_cg=use_reduced_cg,
                dtype=dtype,
            )[-1]
            path_weight.append(not torch.equal(u_matrix, torch.zeros_like(u_matrix)))
            self.register_buffer(f"U_matrix_{nu}", u_matrix)

        self.contractions_weighting = torch.nn.ModuleList()
        self.contractions_features = torch.nn.ModuleList()
        self.weights = torch.nn.ParameterList([])

        for i in range(self.correlation, 0, -1):
            num_params = self.U_tensors(i).size()[-1]
            num_equivariance = 2 * irrep_out.lmax + 1
            num_ell = self.U_tensors(i).size()[-2]

            if i == self.correlation:
                parse_subscript_main = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                    + ["ik,ekc,bci,be -> bc"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                )
                graph_module_main = torch.fx.symbolic_trace(
                    lambda x, y, w, z: torch.einsum(
                        "".join(parse_subscript_main), x, y, w, z
                    )
                )
                self.graph_opt_main = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_main,
                    example_inputs=(
                        torch.randn(
                            [num_equivariance] + [num_ell] * i + [num_params]
                        ).squeeze(0),
                        torch.randn((num_elements, num_params, self.num_features)),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                self.weights_max = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                )
            else:
                parse_subscript_weighting = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                    + ["k,ekc,be->bc"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                )
                parse_subscript_features = (
                    ["bc"]
                    + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                    + ["i,bci->bc"]
                    + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                )

                graph_module_weighting = torch.fx.symbolic_trace(
                    lambda x, y, z: torch.einsum(
                        "".join(parse_subscript_weighting), x, y, z
                    )
                )
                graph_module_features = torch.fx.symbolic_trace(
                    lambda x, y: torch.einsum("".join(parse_subscript_features), x, y)
                )
                graph_opt_weighting = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_weighting,
                    example_inputs=(
                        torch.randn(
                            [num_equivariance] + [num_ell] * i + [num_params]
                        ).squeeze(0),
                        torch.randn((num_elements, num_params, self.num_features)),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                graph_opt_features = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_features,
                    example_inputs=(
                        torch.randn(
                            [BATCH_EXAMPLE, self.num_features, num_equivariance]
                            + [num_ell] * i
                        ).squeeze(2),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                    ),
                )
                self.contractions_weighting.append(graph_opt_weighting)
                self.contractions_features.append(graph_opt_features)
                self.weights.append(
                    torch.nn.Parameter(
                        torch.randn((num_elements, num_params, self.num_features))
                        / num_params
                    )
                )

        for idx, keep in enumerate(path_weight):
            zero_flag = not keep
            if idx < self.correlation - 1:
                if zero_flag:
                    self.weights[idx] = EmptyParam(self.weights[idx])
                self.register_buffer(
                    f"weights_{idx}_zeroed",
                    torch.tensor(zero_flag, dtype=torch.bool),
                )
            else:
                if zero_flag:
                    self.weights_max = EmptyParam(self.weights_max)
                self.register_buffer(
                    "weights_max_zeroed",
                    torch.tensor(zero_flag, dtype=torch.bool),
                )

        if not internal_weights:
            self.weights = weights[:-1]
            self.weights_max = weights[-1]

        self._use_scalar_corr3_fast = (
            _USE_SCALAR_CORR3_FAST
            and self.output_lmax == 0
            and self.correlation == 3
            and self.internal_weights
            and self.shared_weights
        )
        if self._use_scalar_corr3_fast:
            self.refresh_scalar_corr3_fast_buffers()

    def refresh_scalar_corr3_fast_buffers(self) -> None:
        """Cache contiguous transposed U matrices used by the scalar corr=3 fast path."""
        u3 = self.U_tensors(3)
        u2 = self.U_tensors(2)
        u1 = self.U_tensors(1)
        num_ell = int(u3.shape[0])
        buffers = {
            "U_matrix_3_fast_t": u3.reshape(num_ell * num_ell, num_ell * u3.shape[-1]).t().contiguous(),
            "U_matrix_2_fast_t": u2.reshape(num_ell * num_ell, u2.shape[-1]).t().contiguous(),
            "U_matrix_1_fast_t": u1.reshape(num_ell, u1.shape[-1]).t().contiguous(),
        }
        for name, value in buffers.items():
            if name in self._buffers:
                setattr(self, name, value)
            else:
                self.register_buffer(name, value, persistent=False)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        if getattr(self, "_use_scalar_corr3_fast", False):
            self.refresh_scalar_corr3_fast_buffers()

    def _forward_scalar_corr3(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        channels = x.shape[1]
        num_ell = x.shape[2]
        num_elements = y.shape[-1]

        u3 = self.U_tensors(3)
        u2 = self.U_tensors(2)
        u1 = self.U_tensors(1)

        w3 = torch.matmul(y, self.weights_max.reshape(num_elements, -1))
        w3 = w3.view(batch, u3.shape[-1], channels).permute(0, 2, 1).contiguous()
        w2 = torch.matmul(y, self.weights[0].reshape(num_elements, -1))
        w2 = w2.view(batch, u2.shape[-1], channels).permute(0, 2, 1).contiguous()
        w1 = torch.matmul(y, self.weights[1].reshape(num_elements, -1))
        w1 = w1.view(batch, u1.shape[-1], channels).permute(0, 2, 1).contiguous()

        z3 = (x.unsqueeze(-1) * w3.unsqueeze(-2)).reshape(
            batch * channels, num_ell * u3.shape[-1]
        )
        out2 = torch.matmul(
            z3,
            self.U_matrix_3_fast_t.to(x),
        ).view(batch, channels, num_ell, num_ell)

        c2 = torch.matmul(
            w2.reshape(batch * channels, u2.shape[-1]),
            self.U_matrix_2_fast_t.to(x),
        ).view(batch, channels, num_ell, num_ell)
        out1 = torch.sum((out2 + c2) * x.unsqueeze(-2), dim=-1)

        c1 = torch.matmul(
            w1.reshape(batch * channels, u1.shape[-1]),
            self.U_matrix_1_fast_t.to(x),
        ).view(batch, channels, num_ell)
        out = torch.sum((out1 + c1) * x, dim=-1)
        return out.reshape(batch, -1)

    def _forward_selected_elements(
        self,
        x: torch.Tensor,
        node_type_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Select the active element weights before the MACE recursion.

        The original ``y`` argument is one-hot and only selects the leading
        element axis.  Material models can carry O(100) elements, so performing
        that selection as a dense contraction is needlessly expensive.
        ``index_select`` is algebraically identical, including parameter
        gradients: repeated species rows are accumulated by autograd.
        """
        idx = node_type_idx.to(device=x.device, dtype=torch.long)
        if getattr(self,"_use_banked_contraction",False):
            from .banked_contraction import banked_contraction
            banked=banked_contraction(self,x,idx)
            if banked is not None:return banked
        if getattr(self,"_use_species_bank",False):
            from .species_polynomial import contract_species_bank
            fast=contract_species_bank(self,x,idx)
            if fast is not None:return fast

        def selected_coefficient(
            order: int,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            selected = weight.to(dtype=x.dtype, device=x.device).index_select(0, idx)
            u_tensor = self.U_tensors(order).to(dtype=x.dtype, device=x.device)
            if self.output_lmax == 0:
                if order == 1:
                    return torch.einsum("ip,bpc->bci", u_tensor, selected)
                if order == 2:
                    return torch.einsum("ijp,bpc->bcij", u_tensor, selected)
                if order == 3:
                    return torch.einsum("ijkp,bpc->bcijk", u_tensor, selected)
            else:
                if order == 1:
                    return torch.einsum("mip,bpc->bcmi", u_tensor, selected)
                if order == 2:
                    return torch.einsum("mijp,bpc->bcmij", u_tensor, selected)
                if order == 3:
                    return torch.einsum("mijkp,bpc->bcmijk", u_tensor, selected)
            # [N,P,C] x [...,P] -> [N,C,...], retaining every polynomial
            # input axis and the optional output-component axis. Lower orders
            # keep their established expressions and reduction order.
            return torch.tensordot(selected,u_tensor,dims=([1],[u_tensor.ndim-1]))

        out = selected_coefficient(self.correlation, self.weights_max)
        for order, weight in zip(
            range(self.correlation - 1, 0, -1),
            self.weights,
        ):
            out = torch.einsum("bc...i,bci->bc...", out, x)
            out = out + selected_coefficient(order, weight)
        out = torch.einsum("bc...i,bci->bc...", out, x)
        return out.reshape(out.shape[0], self.num_features * (2 * self.output_lmax + 1))

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        node_type_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if node_type_idx is not None:
            return self._forward_selected_elements(x, node_type_idx)
        if self._use_scalar_corr3_fast and x.dtype == torch.float32:
            return self._forward_scalar_corr3(x, y)
        out = self.graph_opt_main(
            self.U_tensors(self.correlation).to(x),
            self.weights_max,
            x,
            y,
        )
        for i, (weight, contract_weights, contract_features) in enumerate(
            zip(self.weights, self.contractions_weighting, self.contractions_features)
        ):
            c_tensor = contract_weights(
                self.U_tensors(self.correlation - i - 1).to(x),
                weight,
                y,
            )
            c_tensor = c_tensor + out
            out = contract_features(c_tensor, x)
        return out.view(out.shape[0], -1)

    def U_tensors(self, nu: int) -> torch.Tensor:
        return dict(self.named_buffers())[f"U_matrix_{nu}"]


class EmptyParam(torch.nn.Parameter):
    def __new__(cls, data):  # pylint: disable=signature-differs
        zero = torch.zeros_like(data)
        return super().__new__(cls, zero, requires_grad=False)

    def requires_grad_(self, mode: bool = True):  # pylint: disable=arguments-differ
        del mode
        return self
