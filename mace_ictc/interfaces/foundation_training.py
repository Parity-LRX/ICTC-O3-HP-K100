# SPDX-License-Identifier: MIT
"""MH-1 training through ICTC kernels with canonical MACE parameters kept live.

No inference embedding/projection cache is used as a trainable substitute.
The original parameter objects own every learned value, permitting a direct
parameter-gradient comparison and an ordinary native MACE checkpoint export.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F
from mace_ictc.interfaces.foundation_hp import FoundationConvolution, FoundationProduct, _blocks
from mace_ictc.interfaces.mace_converter import _calibrate_conv_tp, _mace_conv_tp_paths
from mace_ictc.interfaces.foundation_energy import foundation_forward_fp64
from mace_ictc.models.foundation_linear import linear_plan, node_component_linear, projected_species_input


@dataclass(frozen=True)
class FoundationTrainingConfig:
    """Explicit MH-1 schedules; defaults retain the qualified original route."""
    training_gate_schedule: str | tuple[str, ...] = "recompute"
    training_message_linear_layout: str = "native"
    radial_input: str = "concatenated"
    message_chunk_edges: int = 32768
    mixed_channel_tile: int | None = None
    training_adjoint_channel_tile: int | None = None
    mixed_geometry_schedule: str = "fused"
    gate_gemm_rows: int = 0
    layout: str = "fused_pack"

    def __post_init__(self):
        if isinstance(self.training_gate_schedule, list):
            object.__setattr__(self, "training_gate_schedule", tuple(self.training_gate_schedule))

    @classmethod
    def k100_mh1(cls, *, mode="speed"):
        """Measured two-layer MH-1 recipes; other shapes/stacks need qualification."""
        if mode not in ("speed", "memory"):
            raise ValueError("K100 MH-1 mode must be speed or memory")
        return cls(radial_input="projected_species",
                   training_gate_schedule="materialized" if mode == "speed" else "recompute")

    def validate(self, depth):
        if not isinstance(self.training_gate_schedule, (str, tuple)):
            raise ValueError("Gate schedules must be a string or a per-interaction sequence")
        schedules = ((self.training_gate_schedule,) * depth if isinstance(self.training_gate_schedule, str)
                     else tuple(self.training_gate_schedule))
        if len(schedules) != depth or any(s not in ("recompute", "cached", "materialized") for s in schedules):
            raise ValueError("Supply one valid gate schedule per interaction")
        if self.training_message_linear_layout not in ("native", "node_component"):
            raise ValueError("MH-1 linear layout must be native or node_component")
        if self.radial_input not in ("concatenated", "projected_species"):
            raise ValueError("Radial input must be concatenated or projected_species")
        if self.layout not in ("packed", "fused_pack"):
            raise ValueError("MH-1 layout must be packed or fused_pack")
        if isinstance(self.message_chunk_edges, bool) or not isinstance(self.message_chunk_edges, int) or self.message_chunk_edges < 1:
            raise ValueError("message_chunk_edges must be a positive integer")
        if isinstance(self.gate_gemm_rows, bool) or not isinstance(self.gate_gemm_rows, int) or self.gate_gemm_rows < 0:
            raise ValueError("gate_gemm_rows must be a nonnegative integer")
        for value in (self.mixed_channel_tile, self.training_adjoint_channel_tile):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value not in (0, 32, 64)):
                raise ValueError("Channel tiles must be None, 0, 32 or 64")
        if self.mixed_geometry_schedule not in ("fused", "split"):
            raise ValueError("Mixed geometry schedule must be fused or split")
        if any(s != "materialized" for s in schedules) and (
                self.training_adjoint_channel_tile is not None or self.mixed_geometry_schedule != "fused"):
            raise ValueError("First-adjoint tiling and split geometry require materialized gates")
        return schedules


class LiveGate(nn.Linear):
    """A differentiable change of basis of the original final radial linear."""
    def __init__(self, original, order, scales):
        nn.Module.__init__(self)
        object.__setattr__(self, "original", original)
        self.in_features = original.in_features + 1
        self.out_features = original.out_features
        self.register_buffer("order", order)
        self.register_buffer("scales", scales)

    @property
    def weight(self):
        layer = self.original
        bias = layer.weight.new_zeros(layer.out_features, 1) if layer.bias is None else layer.bias[:, None]
        return torch.cat((layer.weight, bias), 1).index_select(0, self.order) * self.scales[:, None]

    @property
    def bias(self):
        return None


class LiveConvolution(FoundationConvolution):
    def __init__(self, original, species_count):
        super().__init__(original, species_count)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        paths = _mace_conv_tp_paths(original.conv_tp)
        _, scales = _calibrate_conv_tp(original.conv_tp, self.tp,
            input_lmax=self.input_lmax, lmax=self.edge_lmax, channels=self.channels)
        order = torch.tensor([paths.index(path) * self.channels + channel
            for path in self.tp.paths for channel in range(self.channels)])
        scale = torch.tensor([scales[path] for path in self.tp.paths
            for _ in range(self.channels)], dtype=torch.float64)
        del self.radial_hidden, self.density_fn, self.projection
        self.projection = nn.Sequential(nn.Identity(), LiveGate(original.conv_tp_weights.net[-1], order, scale))
        self.register_buffer("species_identity", torch.eye(species_count, dtype=torch.float64))
        object.__setattr__(self, "original", original)
        self.native_paths = paths

    def configure_training(self, config, schedule):
        self.generated = True
        self.tp.internal_compute_dtype = torch.float32
        options = dict(_generated_forward=True, _generated_adjoint=True,
            _generated_specialized_adjoint=True, _generated_second=True,
            _generated_second_roles=True, _generated_layout=config.layout,
            _message_chunk_edges=config.message_chunk_edges, _generated_csr_forward=True,
            _generated_topology="native", _csr_slice_search=True,
            _csr_accumulator_limit=24, _generated_geometry_streaming=True,
            _training_gate_schedule=schedule, _gate_cache_bytes=0,
            _generated_second_channel_tile=config.mixed_channel_tile,
            _training_adjoint_channel_tile=config.training_adjoint_channel_tile,
            _generated_second_harmonic_split=config.mixed_geometry_schedule == "split",
            _training_gate_gemm_rows=config.gate_gemm_rows)
        for name, value in options.items():
            setattr(self.tp, name, value)
        self.config = config
        self.linear_plan = linear_plan(self.original.linear_1)

    def forward(self, projected, harmonics, radial, cutoff, species, src, dst, owned):
        if owned != species.numel():
            raise ValueError("Training uses complete graphs per data-parallel rank")
        original = self.original
        source = original.source_embedding(self.species_identity)
        target = original.target_embedding(self.species_identity)
        # Dense embedding backward sorts/reduces repeated species indices. The
        # generic advanced-index backward has severe atomic contention on K100.
        if self.config.radial_input == "projected_species":
            net, density_net = original.conv_tp_weights.net, original.density_fn.net
            first = projected_species_input(radial, source, target, species[src], species[dst], net[0])
            hidden = net[1:-1](first)
            density_first = projected_species_input(radial, source, target, species[src], species[dst], density_net[0])
            density_value = density_net[1:](density_first)
        else:
            features = torch.cat((radial, F.embedding(species[src], source),
                                  F.embedding(species[dst], target)), -1)
            hidden = original.conv_tp_weights.net[:-1](features)
            density_value = original.density_fn(features)
        hidden = torch.cat((hidden, torch.ones_like(cutoff)), -1) * cutoff
        edge_density = torch.tanh(density_value.square()) * cutoff
        blocks = {l: value @ getattr(self, f"q_{l}").T
            for l, value in _blocks(projected, self.input_degrees).items()}
        from ictc_o3.ops.streamed_gate_message import streamed_gate_message
        attrs = {l: value.unsqueeze(-2) for l, value in enumerate(harmonics)}
        if self.tp._training_gate_schedule == "materialized":
            from ictc_o3.ops.fused_message import aggregate_message
            last = self.projection[-1]
            gates = F.linear(hidden, last.weight, last.bias)
            outputs = aggregate_message(self.tp, blocks, attrs, gates, src, dst, None, backend="triton_remat")
        else:
            outputs = streamed_gate_message(self.tp, blocks, attrs, hidden, self.projection, src, dst, None)
        if outputs is None:
            raise RuntimeError("ICTC live-weight message training schedule unavailable")
        native_blocks = {l: value @ getattr(self, f"q_{l}") for l, value in outputs.items()}
        pieces = []
        for path in self.native_paths:
            index = self.tp.paths.index(path)
            offset = self.tp.path_offset[index] * self.channels
            pieces.append(native_blocks[path[2]][:, offset:offset + self.channels].flatten(1))
        flat = torch.cat(pieces, -1)
        message = (node_component_linear(flat, original.linear_1, self.linear_plan)
                   if self.config.training_message_linear_layout == "node_component" else original.linear_1(flat))
        density = edge_density.new_zeros((owned, 1)).index_add(0, dst, edge_density)
        self.message_calls += 1
        return message, density


class LiveProduct(FoundationProduct):
    def __init__(self, original, use_reduced_cg):
        super().__init__(original, use_reduced_cg)
        self.requires_grad_(False)
        object.__setattr__(self, "original", original)

    def forward(self, features, attrs, species):
        from ictc_o3.ops.factorized_contraction import _plan
        from ictc_o3.ops.species_circuit import species_circuit
        plan = _plan(self.sc, features)
        if plan is None:
            raise RuntimeError("Actual U constants failed ICTC polynomial reconstruction")
        converted = list(self.sc.symmetric_contractions.contractions)
        originals = list(self.original.symmetric_contractions.contractions)
        banks = []
        for module, order, projection in plan["projections"]:
            original = originals[converted.index(module)]
            weights = [original.weights_max, *original.weights]
            banks.append(torch.einsum("rp,zpc->zcr", projection,
                weights[original.correlation - order].to(features)))
        bank = torch.cat(banks, -1)
        if self.agnostic:
            species = torch.zeros_like(species)
        output = species_circuit(features, bank, species, plan)
        self.calls += 1
        return torch.cat([part.flatten(1) for part in output.split(plan["sizes"], -1)], -1)


class FoundationTrainHP(nn.Module):
    def __init__(self, model, head="omat_pbe", *, config=None):
        super().__init__()
        self.native = copy.deepcopy(model).cpu().double()
        self.head = head
        self.head_index = list(model.heads).index(head)
        self.edge_lmax = int(model.spherical_harmonics._lmax)
        self.convolutions = nn.ModuleList([LiveConvolution(layer, len(model.atomic_numbers))
            for layer in self.native.interactions])
        self.products = nn.ModuleList([LiveProduct(product, bool(getattr(model, "use_reduced_cg", False)))
            for product in self.native.products])
        self.native.requires_grad_(True)
        self._linear_plans = {}
        for index, layer in enumerate(self.native.interactions):
            for name in ("linear_up", "linear_res", "linear_2"):
                self._linear_plans[index, name] = linear_plan(getattr(layer, name))
        self.configure_training(config or FoundationTrainingConfig())
        self.generated = True

    def configure_training(self, config=None, **options):
        """Change schedules without changing parameter objects or checkpoint keys."""
        if config is not None and options:
            raise ValueError("Pass a config or keyword schedules, not both")
        config = config or FoundationTrainingConfig(**options)
        schedules = config.validate(len(self.convolutions))
        from ictc_o3.backends.k100.fp32_policy import strict_fp32
        strict_fp32()
        for layer, schedule in zip(self.convolutions, schedules):
            layer.configure_training(config, schedule)
        self.training_config = config
        self._reference_policy = False
        return dict(selected="generated", architecture="nonlinear MACE foundation", **asdict(config),
                    gate_schedules=list(schedules), energy_accumulation_dtype="float64",
                    canonical_parameters=sum(p.numel() for p in self.native.parameters()))

    def check_training(self):
        sample = next(self.native.parameters())
        reasons = []
        if not sample.is_cuda or sample.dtype != torch.float32:
            reasons.append("Foundation training requires GPU FP32")
        if any(max(layer.input_lmax, layer.edge_lmax) > 3 for layer in self.convolutions):
            reasons.append("Foundation training currently covers hidden/edge L <= 3")
        return dict(supported=not reasons, reasons=reasons)

    def _linear(self, features, index, name):
        linear = getattr(self.native.interactions[index], name)
        return (node_component_linear(features, linear, self._linear_plans[index, name])
                if self.training_config.training_message_linear_layout == "node_component" else linear(features))

    @property
    def heads(self):
        return self.native.heads

    @property
    def atomic_numbers(self):
        return self.native.atomic_numbers

    @property
    def atomic_energies_fn(self):
        return self.native.atomic_energies_fn

    @property
    def scale_shift(self):
        return self.native.scale_shift

    def forward_vectors(self, vectors, species, src, dst):
        if self._reference_policy:
            raise ValueError("The native reference policy accepts complete MACE graph dictionaries through forward")
        native = self.native
        owned = species.numel()
        attrs = F.one_hot(species, len(native.atomic_numbers)).to(vectors)
        lengths = vectors.norm(dim=-1, keepdim=True)
        from ictc_o3 import direction_harmonics
        harmonics = direction_harmonics(vectors / lengths.clamp_min(1e-12), self.edge_lmax, policy="generated")
        radial, cutoff = native.radial_embedding(lengths, attrs, torch.stack((src, dst)), native.atomic_numbers)
        if cutoff is None:
            cutoff = torch.ones_like(lengths)
        heads = torch.full((owned,), self.head_index, device=vectors.device, dtype=torch.long)
        pair = (native.pair_repulsion_fn(lengths, attrs, torch.stack((src, dst)), native.atomic_numbers)
                if hasattr(native, "pair_repulsion") else vectors.new_zeros(owned))
        features = native.node_embedding(attrs)
        energies = [pair]
        for index, (layer, convolution, product, original_product) in enumerate(zip(
                native.interactions, self.convolutions, self.products, native.products)):
            skip = layer.skip_tp(features)
            projected = self._linear(features, index, "linear_up")
            residual = self._linear(projected, index, "linear_res")
            message, density = convolution(projected, harmonics, radial, cutoff, species, src, dst, owned)
            message = message / (density * layer.beta + layer.alpha) + residual
            message = layer.reshape(self._linear(layer.equivariant_nonlin(message), index, "linear_2"))
            features = original_product.linear(product(message, attrs, species))
            if original_product.use_sc:
                features = features + skip
            if len(native.readouts) != 1:
                energies.append(native.readouts[index](features, heads)[torch.arange(owned, device=vectors.device), heads])
        if len(native.readouts) == 1:
            energies.append(native.readouts[0](features, heads)[torch.arange(owned, device=vectors.device), heads])
        # Cast individual terms before any energy sum. Upcasting the reduced
        # FP32 energy cannot recover cancellation or accumulation roundoff.
        node = torch.stack([value.to(torch.float64) for value in energies]).sum(0)
        return native.scale_shift(node, heads)

    def forward(self, data, training=False, compute_force=True, **kwargs):
        if self._reference_policy:
            return foundation_forward_fp64(self.native, data, training=training,
                                           compute_force=compute_force, **kwargs)
        positions = data["positions"]
        src, dst = data["edge_index"]
        species = data["node_attrs"].argmax(-1)
        vectors = (positions[dst] - positions[src] + data["shifts"]).detach().requires_grad_(True)
        node = self.forward_vectors(vectors, species, src, dst).to(torch.float64)
        batch = data["batch"]
        count = data["ptr"].numel() - 1
        interaction = node.new_zeros(count).index_add(0, batch, node)
        result = dict(interaction_energy=interaction)
        if compute_force:
            derivative = torch.autograd.grad(node.sum(), vectors, create_graph=training, retain_graph=training)[0]
            result["forces"] = torch.zeros_like(positions).index_add(0, src, derivative).index_add(0, dst, -derivative)
        return result
