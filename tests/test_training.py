import numpy as np
import pytest
from coati import datasets, Config, fit, load_result

@pytest.mark.parametrize("kind",["gaussian","split","paired"])
def test_fit_reload(kind,tmp_path):
    data=getattr(datasets,kind)(n=40)
    cfg=Config(niters=3,num_samples=32,floor_n_sample=32,median_n_sample=40,hidden_dim=16,sync_loss=kind=="paired",sync_weight=.35 if kind=="paired" else 0.)
    r=fit(data,cfg,output_dir=tmp_path/"run")
    prediction=r.predict()
    assert np.isfinite(prediction["primary"]).all()
    loaded=load_result(tmp_path/"run")
    np.testing.assert_array_equal(prediction["primary"],loaded.predict()["primary"])
    if kind=="paired":
        secondary=r.project_secondary(prediction)
        assert secondary.shape[-1]==3
        np.testing.assert_array_equal(secondary,loaded.project_secondary())
    with pytest.raises(FileExistsError):fit(data,cfg,output_dir=tmp_path/"run")

def test_reject_incompatible_inputs(tmp_path):
    with pytest.raises(ValueError,match="integer multiple"):
        fit(datasets.gaussian(),Config(dt=.3),output_dir=tmp_path/"bad")
    with pytest.raises(ValueError,match="secondary"):
        fit(datasets.gaussian(),Config(sync_loss=True),output_dir=tmp_path/"bad")
    with pytest.raises(ValueError,match="smallest snapshot"):
        fit(datasets.gaussian(n=20),Config(),output_dir=tmp_path/"bad")


def test_normalization_and_accurate_kernel(tmp_path):
    import torch
    data=datasets.paired(n=40)
    torch.save({"scale":2.},tmp_path/"scale.pt")
    r=fit(data,Config(niters=2,num_samples=32,floor_n_sample=32,median_n_sample=40,hidden_dim=16,sync_loss=True,sync_weight=.35,map_type="accurate_kernel"),
          output_dir=tmp_path/"normalized",primal_norm_path=tmp_path/"scale.pt",sec_norm_path=tmp_path/"scale.pt")
    np.testing.assert_allclose(r.predict()["primary"],2*r.predict(raw=False)["primary"])
    np.testing.assert_array_equal(r.project_secondary(),load_result(r.output_dir).project_secondary())


def test_disabled_density_diagnostic(tmp_path):
    r=fit(datasets.paired(n=32),Config(niters=100,num_samples=26,floor_n_sample=26,median_n_sample=32,hidden_dim=8,sync_loss=True,sync_weight=.35,sync_density_loss=False),output_dir=tmp_path/"diagnostic")
    assert np.isfinite(r.predict()["primary"]).all()
