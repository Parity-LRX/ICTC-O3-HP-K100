"""Large repeated-species FP32 affine values and mixed derivatives."""
import copy

import pytest
import torch
from torch.nn import functional as F

from mace_ictc.models.foundation_linear import projected_species_input


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a scheduled GPU allocation")
def test_species_projection_keeps_canonical_forward_and_float64_mixed_reference():
    torch.manual_seed(1030)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"
    rows, species, nr, width, hidden = 4096, 89, 10, 512, 64
    layer = torch.nn.Linear(nr + 2 * width, hidden).to(device)
    reference = copy.deepcopy(layer).double()
    radial = torch.randn(rows, nr, device=device, requires_grad=True)
    source = torch.randn(species, width, device=device, requires_grad=True)
    target = torch.randn(species, width, device=device, requires_grad=True)
    src, dst = torch.arange(rows, device=device) % 8, (torch.arange(rows, device=device) * 3) % 8
    params = (radial, source, target, layer.weight, layer.bias)
    identity = tuple(id(p) for p in layer.parameters())
    state = {key: tensor.detach().clone() for key, tensor in layer.state_dict().items()}
    actual = projected_species_input(radial.sin(), source, target, src, dst, layer)
    with torch.no_grad():
        canonical = layer(torch.cat((radial.sin(), F.embedding(src, source), F.embedding(dst, target)), -1))
    # This fails for the old reassociated first affine, even with TF32 off.
    torch.testing.assert_close(actual.detach(), canonical, rtol=0, atol=0)
    with torch.no_grad():
        inference = projected_species_input(radial.sin(), source, target, src, dst, layer)
    torch.testing.assert_close(inference, canonical, rtol=0, atol=0)

    geometry = radial.detach().double().requires_grad_()
    source64 = source.detach().double().requires_grad_()
    target64 = target.detach().double().requires_grad_()
    expected = reference(torch.cat((geometry.sin(), F.embedding(src, source64), F.embedding(dst, target64)), -1))
    force = torch.autograd.grad(actual.tanh().sum(), radial, create_graph=True)[0]
    force64 = torch.autograd.grad(expected.tanh().sum(), geometry, create_graph=True)[0]
    force_target = torch.randn_like(force)
    energy_target = torch.randn_like(actual)
    force_loss = (force - force_target).square().mean()
    force_loss64 = (force64 - force_target.double()).square().mean()
    force_gradients = torch.autograd.grad(force_loss, params, retain_graph=True)
    force_gradients64 = torch.autograd.grad(force_loss64,
        (geometry, source64, target64, reference.weight, reference.bias), retain_graph=True)
    objective = (actual - energy_target).square().mean() + force_loss
    objective64 = (expected - energy_target.double()).square().mean() + force_loss64
    gradients = torch.autograd.grad(objective, params)
    expected_gradients = torch.autograd.grad(objective64, (geometry, source64, target64, reference.weight, reference.bias))
    for got, want in zip((*force_gradients, *gradients), (*force_gradients64, *expected_gradients)):
        assert float((got.double() - want).norm() / want.norm().clamp_min(1e-12)) < 3e-5
    assert tuple(id(p) for p in layer.parameters()) == identity
    for key, tensor in layer.state_dict().items():
        torch.testing.assert_close(tensor, state[key], rtol=0, atol=0)
