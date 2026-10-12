import numpy as np
import pytest
import torch
from coati import Config, TemporalData, datasets, fit, load_result, train_vector_field
from coati.result import Result
from coati.velocity import VelocityEmbedding, project_velocity, _grid


@pytest.mark.parametrize('multimodal', [False, True])
@pytest.mark.parametrize('balanced', [False, True])
def test_modes_reuse_identical_training(multimodal, balanced, tmp_path):
    cfg = Config(multimodal=multimodal, balanced=balanced, iterations=3, batch_size=26,
                 floor_n_sample=26, median_n_sample=32, hidden_dim=8, map_type='kernel')
    data = datasets.paired(n=32)
    new = train_vector_field(data, cfg, output_dir=tmp_path/'new')
    old = fit(data, Config(extra=cfg.to_dict()), output_dir=tmp_path/'old')
    assert cfg.multimodal is multimodal and cfg.balanced is balanced
    for key, value in new.model.state_dict().items():
        torch.testing.assert_close(value, old.model.state_dict()[key], rtol=0, atol=0)
    np.testing.assert_array_equal(new.predict_trajectory()['primary'], old.predict()['primary'])
    restored = load_result(new.output_dir)
    np.testing.assert_array_equal(new.compute_velocity(), restored.compute_velocity())
    assert new.T_model is None  # explicit kernel or single-modality path


def test_clear_config_and_overrides(tmp_path):
    old = Config(niters=3, lr=.002, num_samples=32)
    named = Config(iterations=3, learning_rate=.002, batch_size=32)
    assert old.to_dict() == named.to_dict()
    for a, b in [('iterations','niters'), ('learning_rate','lr'), ('batch_size','num_samples')]:
        with pytest.raises(ValueError, match='either'):
            Config(**{a:3, b:3})
    with pytest.raises(ValueError, match='conflicts'):
        Config(multimodal=False, sync_loss=True)
    with pytest.raises(TypeError):
        Config(balanced='yes')
    overridden = old.with_parameters(multimodal=True, balanced=False,
                 extra={"alpha_growth":2.,"mass_coefficient":3.,"sync_weight":.6})
    assert overridden.to_dict()["alpha_growth"] == 2.
    assert overridden.to_dict()["mass_coefficient"] == 3.
    assert overridden.to_dict()["sync_weight"] == .6
    changed = old.with_parameters(multimodal=True, balanced=False, iterations=4)
    assert not old.multimodal and old.balanced and old.to_dict()['niters'] == 3
    assert changed.to_dict()['sync_weight'] == .35
    assert changed.to_dict()['mass_coefficient'] == 1.
    custom = Config(multimodal=True, balanced=False, secondary_weight=.7, growth_weight=2., mass_weight=3.)
    assert custom.to_dict()['sync_weight'] == .7
    assert custom.to_dict()['alpha_growth'] == 2.
    assert custom.to_dict()['mass_coefficient'] == 3.
    path=tmp_path/'config.json';custom.save(path)
    assert Config.load(path).to_dict() == custom.to_dict()
    with pytest.raises(ValueError, match='secondary'):
        train_vector_field(datasets.gaussian(n=32), multimodal=True, batch_size=26, output_dir=tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


class LinearField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vector_field_net=torch.nn.Linear(3,2,bias=False).double()
        with torch.no_grad():
            self.vector_field_net.weight.copy_(torch.tensor([[1.,0.,2.],[0.,-1.,3.]], dtype=torch.float64))


@pytest.fixture
def result(tmp_path):
    rng=np.random.default_rng(7)
    data=TemporalData([rng.normal(size=(20,2)),rng.normal(size=(20,2))],[0.,1.])
    return Result(LinearField(),data,{'dimension':2,'primary_scale':2.,'config':{'unbalancedModel':False}},tmp_path)


def test_velocity_units_time_chunking(result):
    x=np.array([[2.,4.],[-2.,6.],[0.,8.]])
    t=np.array([0.,.5,1.])
    expected=np.column_stack([x[:,0]+4*t,-x[:,1]+6*t])
    np.testing.assert_allclose(result.compute_velocity(x,time=t,chunk_size=1),expected)
    np.testing.assert_allclose(result.compute_velocity(x/2,time=t,raw=False),expected/2)
    observed=np.concatenate(result.data.primary);times=np.repeat([0.,1.],20)
    np.testing.assert_allclose(result.compute_velocity(),result.compute_velocity(observed,time=times))
    with pytest.raises(ValueError,match='time'):
        result.compute_velocity(x)
    with pytest.raises(ValueError):
        result.compute_velocity(x,time=[0,1])
    with pytest.raises(ValueError):
        result.compute_velocity(x,time=float('nan'))


def test_projection_recovers_affine_map_and_zero_field():
    rng=np.random.default_rng(8);x=rng.normal(size=(90,3));v=rng.normal(size=x.shape)
    matrix=np.array([[1.,-1.],[2.,.5],[-.3,1.2]])
    y=x@matrix+np.array([4.,-2.])
    field=project_velocity(x,v,y,n_neighbors=10,ridge=0)
    np.testing.assert_allclose(field.velocity,v@matrix,rtol=1e-10,atol=1e-10)
    zeros=project_velocity(x,np.zeros_like(x),y)
    np.testing.assert_array_equal(zeros.velocity,0)
    with pytest.raises(ValueError):
        project_velocity(x,v,y[:-1])


def test_anndata_order_times_and_explicit_storage(result):
    ad=pytest.importorskip('anndata');import pandas as pd
    x=np.concatenate(result.data.primary);times=np.repeat([0.,1.],20)
    matrix=np.array([[2.,-.5],[1.,1.]])
    order=np.random.default_rng(1).permutation(len(x))
    a=ad.AnnData(x[order],obs=pd.DataFrame({'stage':np.array(['early']*20+['late']*20)[order]},index=[str(i) for i in order]))
    a.obsm['X_pca']=x[order];a.obsm['X_umap']=x[order]@matrix
    field=result.compute_velocity_embedding(a,time_key='stage',time_mapping={'early':0.,'late':1.},ridge=0)
    expected=result.compute_velocity(x[order],time=times[order])@matrix
    np.testing.assert_allclose(field.velocity,expected,rtol=1e-10,atol=1e-10)
    assert 'coati_velocity' not in a.obsm
    stored=result.write_velocity(a,copy=True,time_key='stage',time_mapping={'early':0.,'late':1.},ridge=0)
    assert 'coati_velocity' not in a.obsm
    np.testing.assert_allclose(stored.obsm['coati_velocity_umap'],expected)
    for style in ['arrows','grid','stream']:
        ax=result.plot_velocity(a,time_key='stage',time_mapping={'early':0.,'late':1.},color='stage',style=style)
        assert len(ax.collections)>0
        import matplotlib.pyplot as plt;plt.close(ax.figure)
    with pytest.raises(ValueError,match='time_mapping'):
        result.compute_velocity_embedding(a,time_key='stage')


def test_grid_masks_unsupported_region_and_handles_zero():
    rng=np.random.default_rng(3);x=rng.normal(size=(100,2))
    field=VelocityEmbedding(x,np.zeros_like(x))
    _,_,mask=_grid(field,grid_size=21,n_neighbors=20,smooth=1,min_mass=.03)
    assert mask.any() and not mask[0,0] and not mask[-1,-1]
    ax=field.plot(style='stream');assert len(ax.collections)==1
    import matplotlib.pyplot as plt;plt.close(ax.figure)


def test_cli_new_name_defaults_and_plot(tmp_path):
    from coati.cli import main
    import json
    data=datasets.gaussian(n=32)
    keys=data.save_npz(tmp_path/'input')
    out=tmp_path/'cli'
    main(['train-vector-field',str(tmp_path/'input/primary.npz'),'--keys',*keys,
          '--times',*[str(t) for t in data.times],'--iterations','2','--batch-size','26',
          '--balanced','--no-multimodal','--output',str(out)])
    values=json.loads((out/'config.json').read_text())
    assert not values['sync_loss'] and not values['unbalancedModel'] and values['niters']==2
    ad=pytest.importorskip('anndata');import pandas as pd
    x=np.concatenate(data.primary)
    a=ad.AnnData(x,obs=pd.DataFrame({'time':np.repeat(data.times,32)},index=[str(i) for i in range(len(x))]))
    a.obsm['X_pca']=x;a.obsm['X_umap']=x
    a.write_h5ad(tmp_path/'cells.h5ad')
    main(['plot-velocity',str(out),str(tmp_path/'cells.h5ad'),'--style','grid','--output',str(tmp_path/'velocity.png')])
    assert (tmp_path/'velocity.png').stat().st_size>1000
