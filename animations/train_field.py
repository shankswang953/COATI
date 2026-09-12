"""Fit an illustrative neural velocity field to the geometric toy paths.

This is supervised distillation of the geometric paths, not COATI training.
Requires PyTorch only when regenerating the checked-in weights.
"""
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from geometry import STARTS, TERMINAL, solve

torch.set_num_threads(1);torch.manual_seed(23)
t=np.linspace(0,1,65)
paths=np.array([solve(1,a,b)[0] for a,b in zip(STARTS,TERMINAL)])
cs=[CubicSpline(t,p,axis=0) for p in paths]
times=np.linspace(0,1,193)
x=np.concatenate([f(times) for f in cs]);v=np.concatenate([f(times,1) for f in cs]);tt=np.tile(times,5)
inputs=np.column_stack([x/3,2*tt-1]).astype('float32')
targets=(v/6).astype('float32')
model=torch.nn.Sequential(torch.nn.Linear(3,96),torch.nn.Tanh(),torch.nn.Linear(96,96),torch.nn.Tanh(),torch.nn.Linear(96,2))
opt=torch.optim.Adam(model.parameters(),lr=.002)
xt=torch.from_numpy(inputs);yt=torch.from_numpy(targets)
for step in range(10001):
    opt.zero_grad();pred=model(xt);loss=((pred-yt)**2).mean();loss.backward();opt.step()
    if step in [4000,7000]:
        for group in opt.param_groups:group['lr']*=.35
    if step%2000==0:print(step,float(loss.detach()),flush=True)
weights={}
for i,layer in enumerate([model[0],model[2],model[4]]):
    weights[f'w{i}']=layer.weight.detach().numpy();weights[f'b{i}']=layer.bias.detach().numpy()
weights['teacher_paths']=paths
np.savez(Path(__file__).with_name('field_weights.npz'),**weights)
