from pathlib import Path
import numpy as np
import torch
from geometry import STARTS, TERMINAL
from scipy.interpolate import CubicSpline

torch.set_num_threads(1);torch.manual_seed(23)
p=Path(__file__).with_name('field_weights.npz');w=dict(np.load(p))
model=torch.nn.Sequential(torch.nn.Linear(3,96),torch.nn.Tanh(),torch.nn.Linear(96,96),torch.nn.Tanh(),torch.nn.Linear(96,2))
for i,j in enumerate([0,2,4]):
    model[j].weight.data.copy_(torch.from_numpy(w[f'w{i}']));model[j].bias.data.copy_(torch.from_numpy(w[f'b{i}']))
steps=32;dt=1/steps
ref=torch.tensor(np.stack([CubicSpline(np.linspace(0,1,65),v)(np.linspace(0,1,steps+1)) for v in w['teacher_paths']],axis=1),dtype=torch.float32)
start=torch.tensor(STARTS,dtype=torch.float32)
def field(x,t):return 6*model(torch.cat([x/3,torch.full((len(x),1),2*t-1)],dim=1))
opt=torch.optim.Adam(model.parameters(),lr=.000002)
best_loss=float("inf");best=None
for epoch in range(1200):
    opt.zero_grad();x=start;pred=[x]
    for i in range(steps):
        t=i*dt;a=field(x,t);b=field(x+a*dt/2,t+dt/2);c=field(x+b*dt/2,t+dt/2);d=field(x+c*dt,t+dt)
        x=x+dt*(a+2*b+2*c+d)/6;pred.append(x)
    pred=torch.stack(pred)
    loss=((pred-ref)**2).mean()+2*((pred[-1]-ref[-1])**2).mean()
    if float(loss.detach())<best_loss:
        best_loss=float(loss.detach());best={k:v.detach().clone() for k,v in model.state_dict().items()}
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
    if epoch%100==0:print(epoch,float(loss.detach()),flush=True)
    if float(loss.detach())<.00012:break
model.load_state_dict(best)
print("Best rollout loss:",best_loss,flush=True)
for i,j in enumerate([0,2,4]):
    w[f'w{i}']=model[j].weight.detach().numpy();w[f'b{i}']=model[j].bias.detach().numpy()
np.savez(p,**w)
