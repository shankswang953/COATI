"""Check the real loss pipeline against known weighted particle actions."""
from types import SimpleNamespace
import math
import pytest
import torch
from coati import Config
from coati._core.TrainLoss import short_term_loss


class KnownField(torch.nn.Module):
    def __init__(self, growth=0., *, linear=False):
        super().__init__()
        self.speed = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))
        self.growth = torch.nn.Parameter(torch.tensor(growth, dtype=torch.float64))
        self.linear = linear

    def forward(self, time, state):
        x = state[0]
        velocity = self.speed * (x if self.linear else torch.ones_like(x))
        energy = .5 * velocity.square().sum(-1, keepdim=True)
        if len(state) == 2:
            return velocity, energy
        _, logw, _, growth_energy = state
        return (velocity, self.growth.expand_as(logw), energy * logw.exp(),
                torch.zeros_like(growth_energy))


class KnownMap(torch.nn.Module):
    def __init__(self, drift=0.):
        super().__init__()
        self.drift = drift

    def forward(self, x, time):
        return 3 * x + self.drift * time[..., None]


def interval(field, grid, *, weights=None, balanced=False, initial=None,
             previous=None, drift=0.):
    grid = torch.as_tensor(grid, dtype=torch.float64)
    if initial is None:
        initial = torch.zeros((2, 1), dtype=torch.float64)
    count = len(initial)
    if weights is None:
        weights = torch.full((count,), 1 / count, dtype=torch.float64)
    if previous is None:
        previous = dict(primal_x=initial, primal_lnw=weights.log().reshape(-1, 1),
                        secondary_x=3 * initial + drift * grid[0])
    args = SimpleNamespace(**Config(
        multimodal=True, balanced=balanced, batch_size=count, dt=.25,
        sinkhorn_pdf=False, sync_pdf_loss=False, sync_density_loss=False,
        mass_loss=False, density_topk=1,
    ).to_dict())
    args.T_model = KnownMap(drift)
    return short_term_loss(
        field, 0, [float(grid[0]), float(grid[-1])], [initial, initial + 1],
        secondary_data=[3 * initial, 3 * (initial + 1)],
        time_steps_list=[(grid.flip(0), grid)], device=torch.device('cpu'),
        args=args, previous_sample=previous,
    )


@pytest.mark.parametrize('growth', [-math.log(2), 0., math.log(2)])
def test_changing_mass_uses_each_left_endpoint_and_keeps_total_mass(growth):
    field = KnownField(growth)
    weights = torch.tensor([.3, 2.7], dtype=torch.float64)
    losses, _ = interval(field, [0., .25, .5, .75, 1.], weights=weights)
    # Constant mapped speed is 6. Mass begins at 3, rather than being renormalized.
    weighted_time = 1. if growth == 0 else .25 * math.expm1(growth) / math.expm1(.25 * growth)
    expected = 18 * 3 * weighted_time
    assert losses['sync_secondary_energy_loss'].item() == pytest.approx(expected, rel=1e-6)
    torch.testing.assert_close(losses['sync_secondary_energy_loss'],
                               9 * losses['forward_energy_velocity'], rtol=1e-6, atol=1e-8)


def test_balanced_formula_is_unchanged_and_uniform_unbalanced_matches():
    field = KnownField()
    grid = [0., .25, .5, .75, 1.]
    balanced, _ = interval(field, grid, balanced=True)
    unbalanced, _ = interval(field, grid)
    assert balanced['sync_secondary_energy_loss'].item() == pytest.approx(18.)
    torch.testing.assert_close(balanced['sync_secondary_energy_loss'], unbalanced['sync_secondary_energy_loss'])


def test_energy_differentiates_through_growth_and_particle_mass():
    field = KnownField()
    initial_logw = torch.tensor([.3, 2.7], dtype=torch.float64).log().requires_grad_()
    previous = dict(primal_x=torch.zeros((2, 1), dtype=torch.float64),
                    primal_lnw=initial_logw[:, None], secondary_x=torch.zeros((2, 1), dtype=torch.float64))
    losses, _ = interval(field, [0., .25, .5, .75, 1.], previous=previous)
    losses['sync_secondary_energy_loss'].backward()
    assert field.growth.grad.item() == pytest.approx(54 * .375, rel=1e-6)
    assert field.speed.grad.item() == pytest.approx(54., rel=1e-6)
    torch.testing.assert_close(initial_logw.grad, 18 * initial_logw.detach().exp())


def test_particle_specific_energy_and_actual_step_lengths():
    field = KnownField(math.log(2), linear=True)
    initial = torch.tensor([[1.], [2.], [3.]], dtype=torch.float64)
    weights = torch.tensor([.1, .2, .7], dtype=torch.float64)
    # The low-level solver grid may differ from configured dt; use its actual intervals.
    losses, _ = interval(field, [0., .25, .5, 1.], initial=initial, weights=weights)
    # For Y=3X, both weighted actions must differ by exactly the factor 9.
    torch.testing.assert_close(losses['sync_secondary_energy_loss'],
                               9 * losses['forward_energy_velocity'], rtol=1e-6, atol=1e-8)


def test_time_dependent_map_includes_mapped_motion_with_zero_primary_velocity():
    field = KnownField()
    with torch.no_grad():
        field.speed.zero_()
    losses, _ = interval(field, [0., .25, .5, .75, 1.], drift=2.)
    assert losses['forward_energy_velocity'].item() == 0
    assert losses['sync_secondary_energy_loss'].item() == pytest.approx(2.)


def test_later_interval_keeps_the_carried_mass():
    field = KnownField(math.log(2))
    first, carried = interval(field, [0., .25, .5])
    second, _ = interval(field, [.5, .75, 1.], previous=carried)
    whole, _ = interval(field, [0., .25, .5, .75, 1.])
    torch.testing.assert_close(first['sync_secondary_energy_loss'] + second['sync_secondary_energy_loss'],
                               whole['sync_secondary_energy_loss'], rtol=1e-6, atol=1e-8)
