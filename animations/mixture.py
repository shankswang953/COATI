"""Contours of summed Gaussian densities, not superimposed component ellipses."""
import numpy as np
from contourpy import contour_generator
from geometry import INITIAL, TERMINAL, SHAPES

def mixture_contours(means,shapes):
    xx=np.linspace(-3.6,3.8,520);yy=np.linspace(-3.0,3.2,450)
    x,y=np.meshgrid(xx,yy);density=np.zeros_like(x)
    for (mx,my),(sx,sy,a) in zip(means,shapes):
        sx*=1.45;sy*=1.45
        dx=x-mx;dy=y-my
        u=dx*np.cos(a)+dy*np.sin(a);v=-dx*np.sin(a)+dy*np.cos(a)
        density+=np.exp(-.5*((u/sx)**2+(v/sy)**2))/(2*np.pi*sx*sy*len(means))
    gen=contour_generator(x=xx,y=yy,z=density)
    peak=density.max()
    # Highest low-density contour that encloses the whole mixture in one component.
    outer=None
    for level in np.geomspace(.20,.001,70)*peak:
        lines=gen.lines(level)
        if len(lines)==1 and np.linalg.norm(lines[0][0]-lines[0][-1])<1e-5:
            outer=level;break
    if outer is None:raise ValueError('Mixture outer contour did not merge.')
    levels=[outer,min(.4*peak,max(outer*2.8,.19*peak)),.64*peak]
    return [gen.lines(level) for level in levels],levels

if __name__=='__main__':
    for means,shapes in [(INITIAL,SHAPES[:2]),(TERMINAL,SHAPES[2:])]:
        rings,levels=mixture_contours(means,shapes)
        assert len(rings)==3 and len(rings[0])==1
        print('Mixture components:',len(means),'contour loops per level:',list(map(len,rings)))
