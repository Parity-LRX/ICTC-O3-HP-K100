"""Inference adapter for the nonlinear MACE foundation architecture.

The original radial embedding, density normalization, nonlinearities, residual
linears and all readout heads are retained. Convolution and symmetric products
use calibrated ICTC operators. No learned tensor is approximated or retrained.
"""
from __future__ import annotations

import copy
from collections import Counter
from threading import RLock

import torch
from torch import nn
from torch.nn import functional as F

from mace_ictc.interfaces.mace_converter import (
    _calibrate_conv_tp, _copy_symmetric_contraction,
    _e3nn_linear_effective_matrices, _mace_conv_tp_paths,
    _reorder_linear_input_paths,
)
from ictc_o3.basis.mace import orthogonal_Q_blocks
from ictc_o3.basis.ictd_irreps import EdgeWeightedPathPreservingTensorProduct
from ictc_o3.ops.symmetric_contraction import ICTDBridgeUSymmetricContractionSO3


_LOAD_LOCK = RLock()


def load_foundation_checkpoint(path, *, map_location="cpu"):
    """Load official checkpoints containing legacy e3nn TorchScript buffers.

    Older e3nn stores raw bytes in __codegen__; newer versions tag each buffer
    with its serialization type. Normalize only that container while loading,
    retaining its original TorchScript and every checkpoint tensor verbatim.
    """
    from e3nn.util.codegen import CodeGenMixin
    with _LOAD_LOCK:
        original = CodeGenMixin.__setstate__

        def restore(module, state):
            codegen = state.get("__codegen__")
            if isinstance(codegen, dict) and any(isinstance(value, bytes) for value in codegen.values()):
                state = dict(state)
                state["__codegen__"] = {name: ("torchscript", value) if isinstance(value, bytes)
                    else value for name, value in codegen.items()}
            return original(module, state)

        CodeGenMixin.__setstate__ = restore
        try:
            model = torch.load(path, map_location=map_location, weights_only=False)
        finally:
            CodeGenMixin.__setstate__ = original
    # Metadata added by newer e3nn forwards; the checkpoint operators and
    # tensors stay intact. This matches the qualified A100 foundation loader.
    from e3nn.o3._spherical_harmonics import _spherical_harmonics
    for module in model.modules():
        if module.__class__.__name__ == "SphericalHarmonics" and not hasattr(module, "sph_func"):
            module.sph_func = _spherical_harmonics
        if module.__class__.__name__ == "Activation" and not hasattr(module, "paths"):
            module.paths = [(mul, (ir.l, ir.p), act)
                for (mul, ir), act in zip(module.irreps_in, module.acts)]
    return model


def _degrees(irreps):
    result = {}
    for mul, ir in irreps:
        if ir.p != (-1) ** ir.l or ir.l in result:
            raise ValueError(f"Expected unique natural-parity degree blocks: {irreps}")
        result[ir.l] = int(mul)
    if set(result) != set(range(max(result) + 1)):
        raise ValueError(f"Expected contiguous degrees: {irreps}")
    return result


def _blocks(flat, degrees):
    result, offset = {}, 0
    for degree, channels in degrees.items():
        width = channels * (2 * degree + 1)
        result[degree] = flat[:, offset:offset + width].reshape(-1, channels, 2 * degree + 1)
        offset += width
    if offset != flat.shape[1]:
        raise ValueError("Irrep dimensions do not match the feature tensor")
    return result


class _NarrowRadial(torch.autograd.Function):
    """Keep narrow edge activations; rematerialize frozen radial networks in VJP.

    Element embeddings are fixed lookup tables. Only geometric arguments carry
    derivatives. The native Linear/LayerNorm/SiLU order is retained per edge.
    Higher coordinate derivatives fall back to the full differentiable formula.
    """
    @staticmethod
    def forward(ctx, radial, cutoff, species, src, dst, owner):
        ctx.owner = owner
        ctx.save_for_backward(radial, cutoff, species, src, dst)
        outputs, densities = [], []
        for begin in range(0, src.numel(), owner.radial_chunk):
            end = min(begin + owner.radial_chunk, src.numel())
            h, density = owner.radial_formula(radial[begin:end], cutoff[begin:end],
                species[src[begin:end]], species[dst[begin:end]])
            outputs.append(h)
            densities.append(density)
        if not outputs:
            return radial.new_empty((0, owner.hidden_width + 1)), radial.new_empty((0, 1))
        return torch.cat(outputs), torch.cat(densities)

    @staticmethod
    def backward(ctx, grad_h, grad_density):
        radial, cutoff, species, src, dst = ctx.saved_tensors
        owner = ctx.owner
        higher = torch.is_grad_enabled()
        if higher:
            with torch.enable_grad():
                local_r, local_c = radial.clone(), cutoff.clone()
                h, density = owner.radial_formula(local_r, local_c, species[src], species[dst])
                active = [v for v, need in zip((local_r, local_c), ctx.needs_input_grad[:2]) if need]
                values = iter(torch.autograd.grad((h, density), active, (grad_h, grad_density),
                    create_graph=True, allow_unused=True))
                grads = [next(values) if need else None for need in ctx.needs_input_grad[:2]]
            return *grads, None, None, None, None
        dr = torch.empty_like(radial) if ctx.needs_input_grad[0] else None
        dc = torch.empty_like(cutoff) if ctx.needs_input_grad[1] else None
        for begin in range(0, src.numel(), owner.radial_chunk):
            end = min(begin + owner.radial_chunk, src.numel())
            with torch.enable_grad():
                local_r = radial[begin:end].detach().requires_grad_(ctx.needs_input_grad[0])
                local_c = cutoff[begin:end].detach().requires_grad_(ctx.needs_input_grad[1])
                h, density = owner.radial_formula(local_r, local_c,
                    species[src[begin:end]], species[dst[begin:end]])
                active = [v for v in (local_r, local_c) if v.requires_grad]
                values = iter(torch.autograd.grad((h, density), active,
                    (grad_h[begin:end], grad_density[begin:end]), allow_unused=True))
                for target, value in ((dr, local_r), (dc, local_c)):
                    if value.requires_grad:
                        gradient = next(values)
                        target[begin:end] = torch.zeros_like(value) if gradient is None else gradient
        return dr, dc, None, None, None, None


class FoundationConvolution(nn.Module):
    def __init__(self, interaction, species_count):
        super().__init__()
        self.input_degrees = _degrees(interaction.conv_tp.irreps_in1)
        if len(set(self.input_degrees.values())) != 1:
            raise ValueError("HP TP requires uniform convolution channel multiplicities")
        self.channels = next(iter(self.input_degrees.values()))
        self.input_lmax = max(self.input_degrees)
        self.edge_lmax = interaction.conv_tp.irreps_in2.lmax
        self.output_degrees = _degrees(interaction.linear_1.irreps_out.simplify())
        paths = _mace_conv_tp_paths(interaction.conv_tp)
        self.tp = EdgeWeightedPathPreservingTensorProduct(channels=self.channels,
            lmax=max(self.edge_lmax, max(p[2] for p in paths)), allowed_paths=paths,
            path_policy="full", internal_compute_dtype=torch.float64).double()
        with torch.no_grad():
            self.tp.weight.fill_(1.)
        _, scales = _calibrate_conv_tp(interaction.conv_tp, self.tp,
            input_lmax=self.input_lmax, lmax=self.edge_lmax, channels=self.channels)
        eye = torch.eye(species_count, dtype=torch.float64)
        with torch.no_grad():
            self.register_buffer("source_table", interaction.source_embedding(eye).detach())
            self.register_buffer("target_table", interaction.target_embedding(eye).detach())
        net = interaction.conv_tp_weights.net
        if not isinstance(net[-1], nn.Linear):
            raise ValueError("Expected a final dense radial gate projection")
        self.radial_hidden = copy.deepcopy(net[:-1])
        self.density_fn = copy.deepcopy(interaction.density_fn)
        self.hidden_width = net[-1].in_features
        last = nn.Linear(self.hidden_width + 1, net[-1].out_features, bias=False, dtype=torch.float64)
        self.projection = nn.Sequential(nn.Identity(), last)
        with torch.no_grad():
            for old_index, path in enumerate(paths):
                new_index = self.tp.paths.index(path)
                old_rows = slice(old_index * self.channels, (old_index + 1) * self.channels)
                new_rows = slice(new_index * self.channels, (new_index + 1) * self.channels)
                last.weight[new_rows, :-1] = net[-1].weight[old_rows] * scales[path]
                last.weight[new_rows, -1] = (0. if net[-1].bias is None else
                    net[-1].bias[old_rows] * scales[path])
            self.tp.weight.fill_(1.)
        counts = Counter(p[2] for p in paths)
        matrices = _e3nn_linear_effective_matrices(interaction.linear_1,
            channels_in_by_l={l: self.channels * counts[l] for l in counts},
            channels_out_by_l=self.output_degrees)
        matrices = _reorder_linear_input_paths(matrices, interaction.conv_tp, self.tp,
            channels=self.channels, target_lmax=max(self.output_degrees))
        for degree, matrix in matrices.items():
            self.register_buffer(f"message_matrix_{degree}", matrix.T.contiguous())
        for degree, q in enumerate(orthogonal_Q_blocks(self.edge_lmax)):
            self.register_buffer(f"q_{degree}", q)
        self.radial_chunk = 32768
        self.generated = False
        self.message_calls = 0

    def configure(self, *, chunk_edges=262144, backward_edges=32768, radial_chunk=32768):
        from ictc_o3.backends.k100.fp32_policy import strict_fp32
        strict_fp32()
        if any(p.requires_grad for p in self.parameters()):
            raise ValueError("Foundation streamed inference requires explicitly frozen weights")
        if next(self.parameters()).dtype != torch.float32 or not next(self.parameters()).is_cuda:
            raise ValueError("Foundation HP kernels require FP32 CUDA/HIP")
        self.generated = True
        self.radial_chunk = int(radial_chunk)
        self.tp.internal_compute_dtype = torch.float32
        options = dict(_generated_forward=True, _generated_adjoint=True,
            _generated_specialized_adjoint=True, _generated_second=True,
            _generated_second_roles=True, _generated_layout="fused_pack",
            _message_chunk_edges=int(chunk_edges), _generated_csr_forward=True,
            _generated_topology="native", _csr_slice_search=True,
            _csr_accumulator_limit=24, _generated_geometry_streaming=True,
            _inference_gate_adjoint_rows=16384, _inference_gemm_policy="dtk2604",
            _inference_message_adjoint_edges=int(backward_edges), _gate_cache_bytes=0,
            _inference_degree_layout=True)
        for name, value in options.items():
            setattr(self.tp, name, value)

    def radial_formula(self, radial, cutoff, source_species, target_species):
        features = torch.cat((radial, self.source_table[source_species],
            self.target_table[target_species]), dim=-1)
        hidden = self.radial_hidden(features)
        hidden = torch.cat((hidden, torch.ones_like(cutoff)), dim=-1) * cutoff
        density = torch.tanh(self.density_fn(features).square()) * cutoff
        return hidden, density

    def forward(self, projected, harmonics, radial, cutoff, species, src, dst, owned):
        blocks = {l: value @ getattr(self, f"q_{l}").T
            for l, value in _blocks(projected, self.input_degrees).items()}
        if self.generated:
            hidden, edge_density = _NarrowRadial.apply(radial, cutoff, species, src, dst, self)
            from ictc_o3.ops.streamed_gate_message import streamed_gate_message
            outputs = streamed_gate_message(self.tp, blocks,
                {l: value.unsqueeze(-2) for l, value in enumerate(harmonics)},
                hidden, self.projection, src, dst, None, output_nodes=owned)
            if outputs is None:
                raise RuntimeError("Foundation generated message schedule is unsupported")
        else:
            hidden, edge_density = self.radial_formula(radial, cutoff, species[src], species[dst])
            gates = self.projection(hidden)
            edge_outputs = self.tp({l: value[src] for l, value in blocks.items()},
                {l: value.unsqueeze(-2) for l, value in enumerate(harmonics)}, gates)
            outputs = {l: value.new_zeros(owned, *value.shape[1:]).index_add(0, dst, value)
                for l, value in edge_outputs.items()}
        density = edge_density.new_zeros((owned, 1)).index_add(0, dst, edge_density)
        messages = []
        for degree, value in outputs.items():
            matrix = getattr(self, f"message_matrix_{degree}")
            mixed = F.linear(value.transpose(-2, -1), matrix).transpose(-2, -1)
            messages.append((mixed @ getattr(self, f"q_{degree}")).flatten(1))
        self.message_calls += 1
        return torch.cat(messages, -1), density


class FoundationProduct(nn.Module):
    def __init__(self, product, use_reduced_cg):
        super().__init__()
        source = product.symmetric_contractions
        input_degrees = _degrees(source.irreps_in)
        output_degrees = _degrees(source.irreps_out)
        self.channels = next(iter(input_degrees.values()))
        if set(input_degrees.values()) != {self.channels} or set(output_degrees.values()) != {self.channels}:
            raise ValueError("Expected uniform symmetric-product multiplicities")
        self.lmax, self.target_lmax = max(input_degrees), max(output_degrees)
        elements = source.contractions[0].weights_max.shape[0]
        self.agnostic = bool(getattr(product, "use_agnostic_product", False))
        self.sc = ICTDBridgeUSymmetricContractionSO3(num_elements=elements,
            channels=self.channels, lmax=self.lmax, target_lmax=self.target_lmax,
            correlation=source.contractions[0].correlation, use_reduced_cg=use_reduced_cg).double()
        _copy_symmetric_contraction(source, self.sc)
        # Preserve the checkpoint's actual CG constants, including their stored
        # precision, rather than silently regenerating a different U bank.
        with torch.no_grad():
            for degree, (original, converted) in enumerate(zip(source.contractions,
                    self.sc.symmetric_contractions.contractions)):
                for order in range(1, original.correlation + 1):
                    value = getattr(original, f"U_matrix_{order}")
                    getattr(converted, f"U_matrix_{order}").copy_(
                        self.sc._transform_u_tensor(value, degree))
        for degree, q in enumerate(orthogonal_Q_blocks(max(self.lmax, self.target_lmax))):
            self.register_buffer(f"q_{degree}", q)
        self.calls = 0

    def configure(self, *, native_basis=True):
        from mace_ictc.hp import configure_symmetric_contraction
        plan = configure_symmetric_contraction(self.sc, policy="generated", species_schedule="indexed")
        self.generated_native_basis = bool(native_basis)
        return {**plan, "native_input_basis": self.generated_native_basis}

    def forward(self, features, attrs, species):
        if getattr(self, "generated_native_basis", False):
            # The qualified HP polynomial circuit already evaluates in the
            # native CG basis internally. The preceding original nonlinear
            # layer produces that basis, so eliminate Q.T -> Q and Q.T -> Q
            # round trips and their graph-sized packing buffers altogether.
            # The same actual-U reconstruction gate and learned weights remain.
            from ictc_o3.ops.factorized_contraction import _plan
            from ictc_o3.ops.species_circuit import species_circuit
            plan = _plan(self.sc, features)
            if plan is None:
                raise ValueError("SC constants do not reconstruct within the HP precision budget")
            banks = []
            for module, order, projection in plan["projections"]:
                weights = [module.weights_max, *module.weights]
                banks.append(torch.einsum("rp,zpc->zcr", projection,
                    weights[module.correlation-order].to(features)))
            bank = torch.cat(banks, -1)
            if self.agnostic:
                species = torch.zeros_like(species)
            output = species_circuit(features, bank, species, plan)
            self.calls += 1
            return torch.cat([part.flatten(1) for part in output.split(plan["sizes"], -1)], -1)
        # Native product input is [N,C,(L+1)^2]; bridge-U accepts degree-major flat features.
        blocks = [features[:, :, l*l:(l+1)**2] @ getattr(self, f"q_{l}").T
            for l in range(self.lmax + 1)]
        flat = torch.cat([value.flatten(1) for value in blocks], -1)
        if self.agnostic:
            attrs = attrs.new_ones((attrs.shape[0], 1))
            species = torch.zeros_like(species)
        output = self.sc(flat, attrs, node_type_idx=species)
        self.calls += 1
        return torch.cat([(value @ getattr(self, f"q_{l}")).flatten(1)
            for l, value in _blocks(output,
                {l: self.channels for l in range(self.target_lmax + 1)}).items()], -1)


class FoundationHP(nn.Module):
    """Exact architecture adapter, selecting one original head per evaluation.

    CPU/FP64 uses the differentiable ICTC reference. configure_inference enables
    streamed K100 TP and generated bridge-U symmetric contractions in FP32.
    Distributed calls exchange projected features after linear_up, retaining
    only owned hidden states. All heads remain in the serialized checkpoint.
    """
    def __init__(self, model, *, head="omat_pbe"):
        super().__init__()
        if model.__class__.__name__ != "ScaleShiftMACE":
            raise ValueError("Expected ScaleShiftMACE")
        if any(layer.__class__.__name__ != "RealAgnosticResidualNonLinearInteractionBlock"
            for layer in model.interactions):
            raise ValueError("Foundation adapter supports nonlinear residual interactions")
        if hasattr(model, "joint_embedding"):
            raise ValueError("Additional charge/spin embeddings need an explicit input adapter")
        self.native = copy.deepcopy(model).double().eval()
        self.heads = list(model.heads)
        if head not in self.heads:
            raise ValueError(f"Head {head!r} not found in {self.heads}")
        self.head = head
        self.head_index = self.heads.index(head)
        self.num_interaction = len(model.interactions)
        self.max_radius = float(model.r_max)
        self.edge_lmax = int(model.spherical_harmonics._lmax)
        self.convolutions = nn.ModuleList([FoundationConvolution(layer, len(model.atomic_numbers))
            for layer in self.native.interactions])
        self.products = nn.ModuleList([FoundationProduct(product, bool(getattr(model, "use_reduced_cg", False)))
            for product in self.native.products])
        identity = torch.eye(len(model.atomic_numbers), dtype=torch.float64)
        with torch.no_grad():
            self.register_buffer("embedding_table", self.native.node_embedding(identity).detach())
            self.register_buffer("first_projected_table",
                self.native.interactions[0].linear_up(self.embedding_table).detach())
        for layer in self.native.interactions:
            layer.conv_tp = nn.Identity()
            layer.conv_tp_weights = nn.Identity()
            layer.source_embedding = nn.Identity()
            layer.target_embedding = nn.Identity()
            layer.density_fn = nn.Identity()
            layer.linear_1 = nn.Identity()
        for product in self.native.products:
            product.symmetric_contractions = nn.Identity()
        self.generated = False

    def select_head(self, head):
        """Select one retained original readout head without changing weights."""
        if head not in self.heads:
            raise ValueError(f"Head {head!r} not found in {self.heads}")
        self.head, self.head_index = head, self.heads.index(head)
        return self

    def configure_inference(self, *, chunk_edges=262144, backward_edges=32768, radial_chunk=32768,
                            product_basis="native"):
        if product_basis not in ("native", "bridge"):
            raise ValueError("product_basis must be native or bridge")
        for layer in self.convolutions:
            layer.configure(chunk_edges=chunk_edges, backward_edges=backward_edges, radial_chunk=radial_chunk)
        plans = [product.configure(native_basis=product_basis=="native") for product in self.products]
        self.generated = True
        return dict(message="streamed ICTC generated TP", products=plans,
            chunk_edges=chunk_edges, backward_edges=backward_edges, radial_chunk=radial_chunk,
            product_basis=product_basis,
            layer_temporaries_released=True,
            head=self.head, all_heads_retained=self.heads,
            exchange="projected features; fresh owner map; FP32 forward and adjoint")

    def forward_vectors(self, vectors, species, src, dst, *, owned=None, exchange=None):
        total = species.numel()
        owned = total if owned is None else owned
        attrs = F.one_hot(species, len(self.native.atomic_numbers)).to(vectors)
        lengths = vectors.norm(dim=-1, keepdim=True)
        from ictc_o3 import direction_harmonics
        harmonics = direction_harmonics(vectors / lengths.clamp_min(1e-12), self.edge_lmax,
            policy="generated" if self.generated else "reference")
        radial, cutoff = self.native.radial_embedding(lengths, attrs,
            torch.stack((src, dst)), self.native.atomic_numbers)
        if cutoff is None:
            cutoff = torch.ones_like(lengths)
        node_heads = torch.full((owned,), self.head_index, device=vectors.device, dtype=torch.long)
        if hasattr(self.native, "pair_repulsion"):
            pair = self.native.pair_repulsion_fn(lengths, attrs,
                torch.stack((src, dst)), self.native.atomic_numbers)[:owned]
        else:
            pair = vectors.new_zeros(owned)
        features = self.embedding_table[species[:owned]]
        energies = [pair]
        for index, (native, convolution, product, native_product) in enumerate(zip(
                self.native.interactions, self.convolutions, self.products, self.native.products)):
            sc = native.skip_tp(features)
            projected = native.linear_up(features)
            residual = native.linear_res(projected)
            del features
            if index == 0:
                projected = torch.cat((projected, self.first_projected_table[species[owned:]]), 0)
            elif exchange is not None:
                projected = exchange(index, torch.cat((projected,
                    projected.new_zeros(total - owned, projected.shape[-1])), 0))
            elif owned != total:
                raise ValueError("Distributed hidden features require an exchange callback")
            message, density = convolution(projected, harmonics, radial, cutoff, species, src, dst, owned)
            del projected
            message = message / (density * native.beta + native.alpha) + residual
            del density, residual
            message = native.reshape(native.linear_2(native.equivariant_nonlin(message)))
            contracted = product(message, attrs[:owned], species[:owned])
            del message
            features = native_product.linear(contracted)
            del contracted
            if native_product.use_sc:
                features = features + sc
            del sc
            if len(self.native.readouts) != 1:
                energies.append(self.native.readouts[index](features, node_heads)[
                    torch.arange(owned, device=vectors.device), node_heads])
        if len(self.native.readouts) == 1:
            energies.append(self.native.readouts[0](features, node_heads)[
                torch.arange(owned, device=vectors.device), node_heads])
        interaction = self.native.scale_shift(torch.stack(energies).sum(0), node_heads)
        e0 = self.native.atomic_energies_fn(attrs[:owned])[
            torch.arange(owned, device=vectors.device), node_heads]
        return dict(interaction_energy=interaction, node_energy=e0.double() + interaction.double())
