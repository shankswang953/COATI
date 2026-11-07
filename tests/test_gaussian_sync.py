import numpy as np
import pytest
import torch
import coati as ct
import toy_data as datasets
from gaussian_comparison import make_shared_initial_checkpoint


def test_gaussian_sync_pairing_and_differentiable_known_map():
    data = datasets.gaussian_sync(n=128,seed=7)
    same = datasets.gaussian_sync(n=128,seed=7)
    assert data.times.tolist() == [0.,1.]
    assert all(x.shape == (128,2) for x in data.primary)
    assert all(y.shape == (128,3) for y in data.secondary)
    assert np.linalg.norm(data.primary[-1][:64].mean(0)-[.50,.72]) < .03
    assert np.linalg.norm(data.primary[-1][64:].mean(0)-[.72,.50]) < .03
    odd = datasets.gaussian_sync(n=129, seed=7)
    assert [len(x) for x in odd.primary] == [129,129]
    assert all(((x >= 0.) & (x <= 1.)).all() for x in data.primary)
    mapping = datasets.gaussian_bump_map()
    np.testing.assert_allclose(mapping(torch.tensor([[.5,.5]]))[:,-1].numpy(),[1.])
    for x,y,repeated in zip(data.primary,data.secondary,same.primary):
        np.testing.assert_array_equal(x,repeated)
        np.testing.assert_allclose(mapping(torch.tensor(x)).detach().numpy(),y,rtol=1e-6)
        np.testing.assert_array_equal(y[:,:2],x)
    query = torch.tensor([[.4,.5]],requires_grad=True)
    mapping(query)[...,-1].sum().backward()
    assert torch.isfinite(query.grad).all() and torch.linalg.norm(query.grad)>0
    assert not list(mapping.parameters())
    with pytest.raises(ValueError):datasets.gaussian_sync(n=1)
    with pytest.raises(ValueError):datasets.gaussian_sync(terminal_centers=[[.5,.8]])
    original=datasets.gaussian_sync(terminal_centers=((.5,.8),(.8,.5)))
    np.testing.assert_array_equal(original.primary[0],data.primary[0])
    np.testing.assert_allclose(original.primary[-1]-data.primary[-1],np.repeat([[0.,.08],[.08,0.]],64,axis=0),atol=1e-7)
    with pytest.raises(ValueError):datasets.gaussian_bump_map(width=0)


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


def test_manuscript_identity_kernels_are_truncated_not_compact():
    from scipy.stats import truncnorm
    centers = ((.5,.8),(.8,.5))
    options = dict(width=.08, initial_covariance=np.eye(2),
                   terminal_covariances=np.repeat(np.eye(2)[None],2,axis=0),
                   terminal_centers=centers)
    data = datasets.gaussian_sync(n=5000, seed=7, terminal_mass=2., **options)
    assert [len(x) for x in data.primary] == [5000,10000]
    for sample, center in [(data.primary[0],(.3,.3)),
                           (data.primary[-1][:5000],centers[0]),
                           (data.primary[-1][5000:],centers[1])]:
        expected_mean, expected_variance = [], []
        for mu in center:
            mean, variance = truncnorm.stats(-mu, 1-mu, loc=mu, scale=1., moments='mv')
            expected_mean.append(mean); expected_variance.append(variance)
        np.testing.assert_allclose(sample.mean(0),expected_mean,atol=.015)
        np.testing.assert_allclose(sample.var(0),expected_variance,atol=.006)
        assert ((sample>=0.) & (sample<=1.)).all()
    mapping=datasets.gaussian_bump_map(width=.08)
    np.testing.assert_allclose(mapping(torch.tensor([[.58,.5]]))[:,-1].numpy(),[np.exp(-.5)],rtol=1e-6)
    with pytest.raises(ValueError,match='integer'):
        datasets.gaussian_sync(n=128,terminal_mass=1.001)
    with pytest.raises(ValueError,match='positive definite'):
        datasets.gaussian_sync(initial_covariance=[[1,2],[2,1]])


def test_reference_density_matches_analytic_normalization():
    from scipy.stats import norm
    from gaussian_comparison import gaussian_density_on_unit_square
    query=np.array([[0.,0.],[.3,.3],[.5,.5],[1.,1.],[-.01,.5]])
    mu=np.array([.3,.3])
    norm_constant=np.prod(norm.cdf(1-mu)-norm.cdf(-mu))
    expected=np.exp(-.5*np.square(query-mu).sum(1))/(2*np.pi*norm_constant)
    expected[-1]=0
    np.testing.assert_allclose(gaussian_density_on_unit_square(query,mu,np.eye(2)),expected,rtol=2e-6)


def test_unbalanced_comparison_uses_mass_action_not_path_mean():
    from types import SimpleNamespace
    from gaussian_comparison import compare_gaussian_runs
    times=np.array([0.,.5,1.])
    source=np.array([[0.,0.],[0.,1.]])
    states=source[None]+times[:,None,None]*np.array([1.,0.])
    mapped=np.concatenate([states,np.zeros((*states.shape[:2],1))],axis=-1)
    pred=dict(times=times,primary=states,secondary=mapped,energy=np.zeros((3,2,1)),
              log_weights=np.log(np.array([[.25,.75],[.5,1.5],[1.,3.]]))[...,None])
    pred['energy'][-1,:,0]=[.25,.75]
    result=SimpleNamespace(data=SimpleNamespace(primary=[source,states[-1]]),
        metadata={'primary_scale':1.,'blur_primary':[.05,.05]})
    row=compare_gaussian_runs({'With sync':result},{'With sync':pred}).iloc[0]
    assert row['Primary kinetic energy']==pytest.approx(1.)
    assert row['Secondary kinetic energy']==pytest.approx(.75)
    assert row['Predicted terminal mass']==pytest.approx(4.)
