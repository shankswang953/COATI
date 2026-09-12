"""Restricted geometric action optimization, independent of the training engine."""
import numpy as np
from scipy.optimize import minimize

BUMP_CENTERS=[(0.,0.,2.7,.59),(-.38,.63,.65,.38),(.40,-.62,.65,.38)]
ROT=np.array([[np.cos(.49),-np.sin(.49)],[np.sin(.49),np.cos(.49)]])
INITIAL=np.array([[-2.2,-.48],[-2.2,.48]])@ROT.T
TERMINAL=np.array([[2.2,y] for y in [-1.15,-.57,0,.57,1.15]])@ROT.T
# Distinct points within two initial populations, not branching a single ODE state.
STARTS=np.array([[-2.2,-.58],[-2.2,-.45],[-2.2,-.30],[-2.2,.38],[-2.2,.58]])@ROT.T

def height(x,y):
    return sum(a*np.exp(-((x-mx)**2+(y-my)**2)/(2*s*s)) for mx,my,a,s in BUMP_CENTERS)

def height_gradient(x,y):
    result=np.zeros((len(np.atleast_1d(x)),2))
    for mx,my,a,s in BUMP_CENTERS:
        h=a*np.exp(-((x-mx)**2+(y-my)**2)/(2*s*s))
        result[:,0]-=h*(x-mx)/(s*s);result[:,1]-=h*(y-my)/(s*s)
    return result

def solve(c,start,end,n=65):
    t=np.linspace(0,1,n);base=start[None,:]*(1-t[:,None])+end[None,:]*t[:,None]
    direction=end-start;normal=np.array([-direction[1],direction[0]])/np.linalg.norm(direction)
    dt=1/(n-1)
    def calc(v):
        p=base+np.r_[0,v,0][:,None]*normal
        z=height(p[:,0],p[:,1]);dp=np.diff(p,axis=0);dz=np.diff(z)
        energy=((dp*dp).sum()+c*(dz@dz))/(2*dt)
        flux=np.vstack([np.zeros(2),dp])-np.vstack([dp,np.zeros(2)])
        zflux=np.r_[0,dz]-np.r_[dz,0]
        gradient=(flux+c*zflux[:,None]*height_gradient(p[:,0],p[:,1]))@normal/dt
        return energy,gradient[1:-1]
    candidates=[]
    for a in [-1.7,0,1.7]:
        fit=minimize(calc,a*np.sin(np.pi*t[1:-1]),jac=True,method='L-BFGS-B',options={'maxiter':600,'ftol':1e-12,'gtol':1e-7})
        candidates.append(fit)
    fit=min(candidates,key=lambda r:r.fun)
    return base+np.r_[0,fit.x,0][:,None]*normal,float(fit.fun)

if __name__=='__main__':
    import json
    report=[]
    for start,end in zip(STARTS,TERMINAL):
        p0,e0=solve(0,start,end);p1,e1=solve(1,start,end)
        z0=height(p0[:,0],p0[:,1]);z1=height(p1[:,0],p1[:,1])
        straight=((np.diff(p0,axis=0)**2).sum()+(np.diff(z0)**2).sum())*32
        assert np.allclose(p0,np.linspace(start,end,65),atol=1e-5)
        assert e1<=straight+1e-6
        report.append(dict(straight_secondary_action=float(straight),sync_secondary_action=e1,straight_max_height=float(z0.max()),sync_max_height=float(z1.max())))
    print(json.dumps(report,indent=2))
