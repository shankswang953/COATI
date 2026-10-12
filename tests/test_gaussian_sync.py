import numpy as np
import pytest
import torch
import coati as ct
from coati.toy_comparison import make_shared_initial_checkpoint


def test_gaussian_sync_pairing_and_differentiable_known_map():
    data = ct.datasets.gaussian_sync(n=128,seed=7)
    same = ct.datasets.gaussian_sync(n=128,seed=7)
    assert data.times.tolist() == [0.,1.]
    assert all(x.shape == (128,2) for x in data.primary)
    assert all(y.shape == (128,3) for y in data.secondary)
    mapping = ct.datasets.gaussian_bump_map()
    for x,y,repeated in zip(data.primary,data.secondary,same.primary):
        np.testing.assert_array_equal(x,repeated)
        np.testing.assert_allclose(mapping(torch.tensor(x)).detach().numpy(),y,rtol=1e-6)
        np.testing.assert_array_equal(y[:,:2],x)
    query = torch.tensor([[.1,.3]],requires_grad=True)
    mapping(query)[...,-1].sum().backward()
    assert torch.isfinite(query.grad).all() and torch.linalg.norm(query.grad)>0
    assert not list(mapping.parameters())
    with pytest.raises(ValueError):ct.datasets.gaussian_sync(n=1)
    with pytest.raises(ValueError):ct.datasets.gaussian_bump_map(width=0)


def test_shared_initialization_is_identical_and_preserves_rng(tmp_path):
    cfg=ct.Config(seed=7,hidden_dim=64,n_layers=3,activation='tanh',balanced=True)
    torch.manual_seed(123)
    before=torch.random.get_rng_state().clone()
    a=make_shared_initial_checkpoint(cfg,tmp_path/'a.pt')
    assert torch.equal(torch.random.get_rng_state(),before)
    b=make_shared_initial_checkpoint(cfg,tmp_path/'b.pt')
    state_a=torch.load(a,weights_only=True)['func_state_dict']
    state_b=torch.load(b,weights_only=True)['func_state_dict']
    assert all(torch.equal(state_a[k],state_b[k]) for k in state_a)
    with pytest.raises(FileExistsError):make_shared_initial_checkpoint(cfg,a)
