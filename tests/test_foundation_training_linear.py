"""Original nonlinear MACE linears retain first and mixed derivatives."""
import copy
import itertools

import pytest
import torch
from e3nn import o3

from mace_ictc.models.foundation_linear import linear_plan, node_component_linear, projected_species_input


@pytest.mark.parametrize("options", [
    {"training_gate_schedule": ("cached",)},
    {"training_gate_schedule": "invalid"},
    {"message_chunk_edges": True},
    {"gate_gemm_rows": -1},
    {"mixed_channel_tile": True},
    {"training_gate_schedule": "cached", "mixed_geometry_schedule": "split"},
    {"radial_input": "frozen_table"},
])
def test_foundation_config_rejects_unsupported_schedules(options):
    from mace_ictc.interfaces.foundation_training import FoundationTrainingConfig
    with pytest.raises(ValueError):
        FoundationTrainingConfig(**options).validate(2)


@pytest.mark.parametrize("irreps_in,irreps_out", [
    ("4x0e+3x1o+2x1o+2x2e", "5x0e+4x1o+3x2e+2x3o"),
    ("3x0e+3x1o", "2x0e+2x1o"),
])
def test_canonical_linear_mixed_derivatives(irreps_in, irreps_out):
    torch.manual_seed(83)
    reference = o3.Linear(irreps_in, irreps_out).double()
    live = copy.deepcopy(reference)
    x = torch.randn(7, reference.irreps_in.dim, dtype=torch.float64, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    # Use a nonleaf input to check that upstream geometry remains connected.
    value = node_component_linear(x.sin(), live, linear_plan(live))
    expected = reference(xr.sin())
    torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-12)
    incoming = torch.randn_like(value, requires_grad=True)
    incoming_ref = incoming.detach().clone().requires_grad_()
    actual_first = torch.autograd.grad(value, (x, live.weight), incoming, create_graph=True)
    expected_first = torch.autograd.grad(expected, (xr, reference.weight), incoming_ref, create_graph=True)
    tangents = tuple(torch.randn_like(v) for v in actual_first)
    actual_second = torch.autograd.grad(actual_first, (x, live.weight, incoming), tangents, allow_unused=True)
    expected_second = torch.autograd.grad(expected_first, (xr, reference.weight, incoming_ref), tangents, allow_unused=True)
    for actual, want in zip((*actual_first, *actual_second), (*expected_first, *expected_second)):
        assert (actual is None) == (want is None)
        if want is not None:
            torch.testing.assert_close(actual, want, rtol=1e-11, atol=1e-11)


def test_projected_species_all_parameter_and_mixed_derivatives():
    torch.manual_seed(91)
    layer = torch.nn.Linear(13, 6).double()
    reference = copy.deepcopy(layer)
    leaves = [torch.randn(17, 3, dtype=torch.float64, requires_grad=True),
              torch.randn(5, 4, dtype=torch.float64, requires_grad=True),
              torch.randn(5, 6, dtype=torch.float64, requires_grad=True)]
    right = [v.detach().clone().requires_grad_() for v in leaves]
    src, dst = torch.arange(17) % 5, (torch.arange(17) * 3) % 5
    actual = projected_species_input(leaves[0].sin(), leaves[1], leaves[2], src, dst, layer)
    want = reference(torch.cat((right[0].sin(), right[1][src], right[2][dst]), -1))
    torch.testing.assert_close(actual, want, rtol=1e-12, atol=1e-12)
    left_inputs, right_inputs = [*leaves, *layer.parameters()], [*right, *reference.parameters()]
    first = torch.autograd.grad(actual.square().sum(), left_inputs, create_graph=True)
    first_ref = torch.autograd.grad(want.square().sum(), right_inputs, create_graph=True)
    tangents = [torch.randn_like(v) for v in first]
    second = torch.autograd.grad(first, left_inputs, tangents)
    second_ref = torch.autograd.grad(first_ref, right_inputs, tangents)
    for got, expected in zip((*first, *second), (*first_ref, *second_ref)):
        torch.testing.assert_close(got, expected, rtol=1e-11, atol=1e-11)


def test_paged_gate_gemm_targets_and_tail_rows():
    from mace_ictc.models.training_gate_message import _contract_gate
    torch.manual_seed(813)
    flat = torch.randn(19, 17, dtype=torch.float64)
    hidden = torch.randn(19, 5, dtype=torch.float64)
    weight = torch.randn(17, 5, dtype=torch.float64)
    for use_h, use_a, use_b in itertools.product((False, True), repeat=3):
        dh = torch.full_like(hidden, float("nan")) if use_h else None
        da = torch.ones_like(weight) if use_a else None
        db = torch.ones(17, dtype=torch.float64) if use_b else None
        _contract_gate(flat, hidden, weight, dh, da, db, rows=7)
        if use_h:
            torch.testing.assert_close(dh, flat @ weight)
        if use_a:
            torch.testing.assert_close(da, 1 + flat.T @ hidden)
        if use_b:
            torch.testing.assert_close(db, 1 + flat.sum(0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a scheduled GPU allocation")
def test_generic_backbone_configuration_defaults_unchanged():
    from mace_ictc.hp import create_model, configure_model
    model = create_model(channels=32, lmax=1, num_interactions=1, correlation=2)
    ids = tuple(id(p) for p in model.parameters())
    info = configure_model(model, training=True, policy="generated")
    assert info["selected"] == "generated"
    assert info["training_gate_schedule"] == "materialized"
    assert info["training_message_linear_layout"] == "node_component"
    assert tuple(id(p) for p in model.parameters()) == ids
    assert configure_model(model, training=True, policy="reference")["selected"] == "reference"
