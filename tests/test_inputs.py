import numpy as np
import pandas as pd
import pytest
from coati import TemporalData, Config, datasets

def test_roundtrip_and_adapters(tmp_path):
    data=datasets.paired(n=40)
    keys=data.save_npz(tmp_path)
    loaded=TemporalData.from_npz(tmp_path/"primary.npz",secondary_path=tmp_path/"secondary.npz",keys=keys,times=data.times)
    frames=[]
    for t,x,y in zip(data.times,data.primary,data.secondary):
        frames.append(pd.DataFrame(dict(t=t,x=x[:,0],z=x[:,1],a=y[:,0],b=y[:,1],c=y[:,2])))
    frame=pd.concat(frames,ignore_index=True)
    csv=tmp_path/"data.csv";frame.to_csv(csv,index=False)
    table=TemporalData.from_csv(csv,time_key="t",feature_keys=["x","z"],secondary_keys=["a","b","c"])
    for expected,actual,converted in zip(data.primary,loaded.primary,table.primary):
        np.testing.assert_allclose(expected,actual)
        np.testing.assert_allclose(expected,converted,atol=1e-7)
    ad=pytest.importorskip("anndata")
    a=ad.AnnData(np.concatenate(data.primary),obs=pd.DataFrame({"time":np.repeat(data.times,40)},index=[str(i) for i in range(120)]))
    a.obsm["secondary"]=np.concatenate(data.secondary)
    backed=TemporalData.from_anndata(a,time_key="time",secondary_obsm_key="secondary")
    for x,y in zip(data.secondary,backed.secondary):np.testing.assert_array_equal(x,y)
    a.write_h5ad(tmp_path/"input.h5ad")
    assert len(TemporalData.from_h5ad(tmp_path/"input.h5ad",time_key="time").times)==3

@pytest.mark.parametrize("times", [[0,0],[1,0],[0,float("nan")]])
def test_invalid_times(times):
    with pytest.raises(ValueError): TemporalData([np.ones((32,2))]*2,times)

def test_invalid_pairing_and_config():
    with pytest.raises(ValueError): TemporalData([np.ones((32,2))]*2,[0,1],[np.ones((31,2))]*2)
    with pytest.raises(TypeError): Config(niter=3)
    with pytest.raises(ValueError): Config(niters=0)
    with pytest.raises(ValueError): Config(support_points=True)

def test_tensor_dictionary():
    import torch
    d=TemporalData.from_dict({"a":torch.ones(32,2),"b":torch.zeros(32,2)},times=[0,1])
    assert d.primary[0].dtype==np.float32
