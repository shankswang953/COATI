"""Compare retained training behavior against a pre-cleanup package.

Run with PYTHONPATH pointing to the baseline package and one output argument.
Run again with current src on PYTHONPATH, a fresh output directory, and the
baseline output directory as the second argument. Uses synthetic data only.
"""
from pathlib import Path
import sys,json
import numpy as np
import torch
from coati import Config,MapConfig,datasets,fit,load_result
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
cases=[('balanced','gaussian',{}),('kernel_sync','paired',dict(sync_loss=True,sync_weight=.35,map_type='kernel')),('film_sync','paired',dict(sync_loss=True,sync_weight=.35,map_type='film')),('unbalanced','split',dict(unbalancedModel=True,alpha_growth=1.,mass_loss=True,mass_coefficient=1.)),('adaptive','gaussian',dict(adaptive_lambda=True,extra={'dual_warmup':0,'dual_update_freq':1}))]
report=[]
for name,kind,opts in cases:
 if (out/(name+'.pt')).exists():continue
 cfg=Config(niters=4,num_samples=26,floor_n_sample=26,median_n_sample=32,hidden_dim=8,**opts)
 result=fit(getattr(datasets,kind)(n=32),cfg,output_dir=out/name,map_config=MapConfig(epochs=3,hidden_dim=8) if name=='film_sync' else None)
 pred=result.predict()['primary'];assert np.isfinite(pred).all()
 np.testing.assert_array_equal(pred,load_result(out/name).predict()['primary'])
 torch.save(result.model.state_dict(),out/(name+'.pt'));np.save(out/(name+'.npy'),pred)
 if len(sys.argv)>2:
  old=Path(sys.argv[2]);weights=torch.load(old/(name+'.pt'),weights_only=True)
  delta=max(float((v-weights[k]).abs().max()) for k,v in result.model.state_dict().items())
  pred_delta=float(np.max(np.abs(pred-np.load(old/(name+'.npy')))))
  assert delta==0 and pred_delta==0,(name,delta,pred_delta)
  report.append(dict(case=name,max_parameter_difference=delta,max_trajectory_difference=pred_delta))
 print(name,'passed',flush=True)
(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
