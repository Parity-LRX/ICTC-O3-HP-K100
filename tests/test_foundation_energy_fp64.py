"""Energy sums must preserve small atomic terms without changing parameters."""
import pytest
import torch

from mace_ictc.interfaces.foundation_energy import foundation_forward_fp64


class EnergyFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.atomic_energies_fn = torch.nn.Linear(1, 1, bias=False)
        self.readouts = torch.nn.ModuleList([torch.nn.Linear(1, 1, bias=False) for _ in range(3)])
        with torch.no_grad():
            self.atomic_energies_fn.weight.fill_(1e8)
            for module, value in zip(self.readouts, (1e8, 1., -1e8)):
                module.weight.fill_(value)

    def forward(self, data):
        values = [module(data["x"].square()) for module in self.readouts]
        if data.get("fail", False):
            raise RuntimeError("fixture forward failure")
        node = torch.stack(values).sum(0).flatten() * 1.75 + .25
        interaction = node.new_zeros(1).index_add(0, data["batch"], node)
        e0 = self.atomic_energies_fn(torch.ones_like(data["x"])).flatten()
        e0 = e0.new_zeros(1).index_add(0, data["batch"], e0)
        return dict(interaction_energy=interaction, energy=e0 + interaction)


def test_native_energy_promotes_terms_before_cancelling_layer_sum():
    model = EnergyFixture().float()
    data = dict(x=torch.tensor([[1.], [2.], [3.]], requires_grad=True),
                batch=torch.zeros(3, dtype=torch.long))
    ids = tuple(id(p) for p in model.parameters())
    keys = tuple(model.state_dict())
    hooks = tuple(len(module._forward_hooks) for module in model.modules())
    ordinary = model(data)
    assert ordinary["interaction_energy"].dtype == torch.float32
    assert abs(float(ordinary["interaction_energy"].detach()) - 25.25) > 1.
    value = foundation_forward_fp64(model, data)
    assert value["interaction_energy"].dtype == value["energy"].dtype == torch.float64
    torch.testing.assert_close(value["interaction_energy"], torch.tensor([25.25], dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(value["energy"], torch.tensor([300000025.25], dtype=torch.float64), rtol=0, atol=0)
    parameters = [module.weight for module in model.readouts]
    gradients = torch.autograd.grad(value["interaction_energy"].sum(), parameters, create_graph=True)
    for gradient in gradients:
        assert gradient.dtype == torch.float32
        torch.testing.assert_close(gradient, torch.tensor([[24.5]]), rtol=0, atol=0)
    mixed = torch.autograd.grad(sum(g.sum() for g in gradients), data["x"])[0]
    torch.testing.assert_close(mixed, 10.5 * data["x"], rtol=0, atol=0)
    assert ids == tuple(id(p) for p in model.parameters())
    assert keys == tuple(model.state_dict())
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert hooks == tuple(len(module._forward_hooks) for module in model.modules())


def test_native_energy_hooks_are_removed_when_forward_raises():
    model = EnergyFixture().float()
    hooks = tuple(len(module._forward_hooks) for module in model.modules())
    with pytest.raises(RuntimeError, match="fixture forward failure"):
        foundation_forward_fp64(model, dict(x=torch.ones(3, 1), batch=torch.zeros(3, dtype=torch.long), fail=True))
    assert hooks == tuple(len(module._forward_hooks) for module in model.modules())
