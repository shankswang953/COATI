"""NumPy inference and RK4 integration of the fitted illustrative neural field."""
from pathlib import Path
import numpy as np
from geometry import STARTS, TERMINAL

class NeuralField:
    def __init__(self):
        self.weights=dict(np.load(Path(__file__).with_name('field_weights.npz')))
    def __call__(self,x,t):
        x=np.atleast_2d(x);a=np.column_stack([x/3,np.full(len(x),2*t-1)])
        for i in range(3):
            a=a@self.weights[f'w{i}'].T+self.weights[f'b{i}']
            if i<2:a=np.tanh(a)
        return 6*a
    def integrate(self,starts=STARTS,n=257):
        dt=1/(n-1);x=np.array(starts).copy();out=[x.copy()]
        for i in range(n-1):
            t=i*dt;k1=self(x,t);k2=self(x+dt*k1/2,t+dt/2)
            k3=self(x+dt*k2/2,t+dt/2);k4=self(x+dt*k3,t+dt)
            x=x+dt*(k1+2*k2+2*k3+k4)/6;out.append(x.copy())
        return np.array(out)

if __name__=='__main__':
    field=NeuralField();p=field.integrate()
    errors=np.linalg.norm(p[-1]-TERMINAL,axis=1)
    print('RK4 endpoint errors:',errors.tolist())
    assert errors.max()<.18, 'Illustrative neural field misses a target cluster.'
