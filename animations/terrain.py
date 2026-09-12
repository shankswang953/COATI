"""Smooth scientific surface rendering for Manim's fixed camera."""
from pathlib import Path
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.transforms import IdentityTransform
from PIL import Image
from geometry import height, height_gradient

def render_surface(path):
    path=Path(path)
    if path.exists():return path
    path.parent.mkdir(parents=True,exist_ok=True)
    xs=np.linspace(-2.9,2.9,181);ys=np.linspace(-2.5,2.5,151)
    x,y=np.meshgrid(xs,ys);z=height(x,y)
    sx=3.55+.90*x+.12*y;sy=-.33+.68*y+.58*z
    points=np.stack([(sx-.4)/6.5*1300,(sy+2.05)/4*800],axis=-1).reshape(-1,2)
    grad=height_gradient(x.ravel(),y.ravel()).reshape(*x.shape,2)
    norm=np.dstack([-grad,np.ones_like(z)]);norm/=np.linalg.norm(norm,axis=2,keepdims=True)
    light=np.array([-.5,-.65,1.]);light/=np.linalg.norm(light)
    diffuse=np.clip(norm@light,0,1)
    base=np.array([.98,.57,.17]);pale=np.array([1.,.92,.79])
    amount=np.clip(z/2.4,.05,1)[...,None]
    rgb=pale*(1-amount)+base*amount
    rgb*= (.80+.20*diffuse[...,None])
    rgb+= .055*np.power(diffuse[...,None],10)
    alpha=np.clip(z/.13,0,1)
    rgb=rgb*alpha[...,None]+np.array([250,251,253])/255*(1-alpha[...,None])
    colors=np.dstack([np.clip(rgb,0,1),np.ones_like(alpha)]).reshape(-1,4)
    triangles=[]
    for j in range(len(ys)-2,-1,-1):
        for i in range(len(xs)-1):
            a=j*len(xs)+i;b=a+1;c=a+len(xs);d=c+1
            triangles.extend([[a,b,d],[a,d,c]])
    tri=np.array(triangles)
    fig=Figure(figsize=(13,8),dpi=100,facecolor=(0,0,0,0));canvas=FigureCanvasAgg(fig);canvas.draw()
    renderer=canvas.get_renderer();gc=renderer.new_gc();gc.set_linewidth(0)
    renderer.draw_gouraud_triangles(gc,points[tri],colors[tri],IdentityTransform())
    Image.fromarray(np.asarray(renderer.buffer_rgba())).save(path)
    return path
