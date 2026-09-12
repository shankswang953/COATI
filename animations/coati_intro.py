"""Scientific visual primitives for the COATI film; no training imports."""
from pathlib import Path
import hashlib
import numpy as np
from matplotlib import rc_context
from matplotlib.mathtext import math_to_image
from manim import *

BG = '#FAFBFD'
FG = '#192C43'
MUTED = '#62758A'
BLUE = '#007AC2'
ORANGE = '#F18621'
RED = '#D85059'
GREEN = '#258C70'
GRID = '#DCE5EE'
config.background_color = BG

def words(s, size=25, color=FG):
    # Arial lays out the complete line first; a single path preserves all spacing.
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties, findfont
    from matplotlib.path import Path as MplPath
    folder=Path(config.media_dir)/'arial_text';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(hashlib.sha256((s+str(size)+color+'singlepath-v2').encode()).hexdigest()[:18]+'.svg')
    prop=FontProperties(fname=findfont('Arial',fallback_to_default=False))
    line=TextPath((0,0),s,size=size,prop=prop,usetex=False)
    bounds=line.get_extents();width=bounds.width
    if not path.exists():
        parts=[]
        commands={MplPath.MOVETO:'M',MplPath.LINETO:'L',MplPath.CURVE3:'Q',MplPath.CURVE4:'C'}
        for vertices,code in line.iter_segments(curves=True,simplify=False):
            if code==MplPath.CLOSEPOLY:parts.append('Z');continue
            xy=np.array(vertices).reshape(-1,2);xy[:,1]*=-1
            parts.append(commands[code]+' '.join(f'{v:.5f}' for v in xy.ravel()))
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{bounds.xmin} {-bounds.ymax} {width} {bounds.height}"><path fill="{color}" d="{" ".join(parts)}"/></svg>')
    return SVGMobject(str(path),stroke_width=0).set_width(width*.020)

def formula(s, height=.38, color=FG):
    folder = Path(config.media_dir) / 'formula_svg'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (hashlib.sha256((s+color+'arial-v3').encode()).hexdigest()[:16]+'.svg')
    if not path.exists():
        with rc_context({'savefig.transparent': True, 'font.family': 'Arial', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold', 'mathtext.cal': 'Arial:italic', 'mathtext.sf': 'Arial', 'mathtext.tt': 'Arial', 'mathtext.fallback': 'stixsans'}):
            math_to_image('$'+s+'$', path, format='svg', color=color, dpi=160)
    obj = SVGMobject(str(path)).set_height(height)
    if obj.width > 12.5: obj.scale_to_fit_width(12.5)
    return obj

def curve(points, color=BLUE, width=4):
    return VMobject(stroke_color=color, stroke_width=width).set_points_smoothly(np.array(points))

def bump(center, sx=.36, sy=.24, color=BLUE):
    # Three equal-density contours, with faint fill to distinguish overlapping groups.
    return VGroup(*[Ellipse(width=2*sx*r,height=2*sy*r,stroke_color=color,
        stroke_width=1.5,stroke_opacity=.78,fill_color=color,fill_opacity=.025).move_to(center)
        for r in [3.1,2.0,1.0]])

# Cell-inspired distribution schematics: organic envelopes, membranes and nuclei.
def cell(point,r=.075,color='#93A4B5',angle=0,opacity=.75):
    body=Ellipse(width=2.2*r,height=1.6*r,stroke_color=color,stroke_width=.85,fill_color=color,fill_opacity=.14*opacity).rotate(angle).move_to(point)
    nucleus=Ellipse(width=.65*r,height=.55*r,stroke_width=0,fill_color=color,fill_opacity=.55*opacity).rotate(angle).move_to(np.array(point)+np.array([.12*r,-.06*r,0]))
    return VGroup(body,nucleus)

def population(center,seed=1,n=17,color='#8A9EAF',sx=.48,sy=.31):
    rng=np.random.default_rng(seed)
    angles=np.linspace(0,TAU,90)
    radius=1+.11*np.sin(3*angles+.4*seed)+.065*np.cos(5*angles)
    pts=[np.array(center)+np.array([sx*r*np.cos(a),sy*r*np.sin(a),0]) for a,r in zip(angles,radius)]
    outline=VMobject(stroke_color=color,stroke_width=1,stroke_opacity=.25,fill_color=color,fill_opacity=.07).set_points_smoothly(pts)
    outline.close_path();g=VGroup(outline)
    accepted=[]
    for _ in range(800):
        p=rng.uniform(-1,1,2)
        if (p*p).sum()>.78 or any(np.linalg.norm((p-q)*[sx,sy])<.12 for q in accepted):continue
        accepted.append(p)
        if len(accepted)>=n:break
    for p in accepted:
        q=np.array(center)+np.array([p[0]*sx,p[1]*sy,0])
        g.add(cell(q,.055+rng.uniform(0,.018),color,rng.uniform(-.7,.7)))
    return g

class LossPanels:
    def ablations(self):
        title=words('What keeps a trajectory biologically plausible?',27).move_to([0,3.25,0])
        sub=formula(r'\mathcal{L}=C_x\mathcal{A}_{X}+C_y\mathcal{A}_{Y}+\mathcal{L}_{\mathrm{Ref}}+\mathcal{L}_{\mathrm{Manifold}}+\mathcal{L}_{\mathrm{Mass}}',.31).move_to([0,2.65,0])
        self.play(FadeIn(title),FadeIn(sub),run_time=.6)
        centers=[-4.55,0,4.55];groups=[];failures=[]
        for i,x in enumerate(centers):
            label=words(['Reference matching','Manifold support','Population mass'][i],21).move_to([x,1.87,0])
            eq=formula([r'\mathrm{w/o}\;\mathcal{L}_{\mathrm{Ref}}',r'\mathrm{w/o}\;\mathcal{L}_{\mathrm{Manifold}}',r'\mathrm{w/o}\;\mathcal{L}_{\mathrm{Mass}}'][i],.26,MUTED).move_to([x,1.35,0])
            start=np.array([x-1.35,-.43,0]);mid=np.array([x,.48,0]);end=np.array([x+1.32,-.08,0])
            bg=VGroup(population(start,11,n=13,sx=.48,sy=.36),population(mid,7,n=18,sx=.50,sy=.40),population(end,4,n=19,sx=.54,sy=.41))
            if i==1:
                # A curved cloud of cells shows supported tissue states between snapshots.
                rng=np.random.default_rng(5)
                bridge=curve([start,mid,end],'#B7C5CE',31).set_stroke(opacity=.15)
                bg.add_to_back(bridge)
                for t in np.linspace(.12,.9,25):
                    q=curve([start,mid,end]).point_from_proportion(t)+np.array([0,rng.normal(0,.105),0])
                    bg.add(cell(q,.055,'#A8B7C2',rng.uniform(-.5,.5),.6))
            good=curve([start,mid,end],GREEN,2.5)
            good_cells=VGroup(*[cell(p,.10,GREEN,a) for p,a in zip([start,mid,end],[.2,.6,-.2])])
            g=VGroup(label,eq,bg,good,good_cells)
            if i==0:
                bad_points=[start,[x-.1,-.85,0],[x+1.25,-.90,0]]
                bad=VGroup(curve(bad_points,RED,2.5),population(bad_points[1],16,n=6,color=RED,sx=.24,sy=.17),population(bad_points[2],19,n=6,color=RED,sx=.24,sy=.17))
                caption='Miss observed cell states'
            elif i==1:
                bad_points=[start,[x,-.70,0],end]
                bad=VGroup(curve(bad_points,RED,2.5),*[cell([x+dx,-.69+dy,0],.085,RED,.2) for dx,dy in [(-.26,.01),(.04,-.04),(.32,.01)]])
                caption='Leave the cell manifold'
            else:
                bad=VGroup(curve([start,mid,end],RED,2))
                # Wrong abundance on the same support, rather than arbitrary large circles.
                for k,p in enumerate([mid,end]):
                    rng=np.random.default_rng(31+k)
                    count=20 if k==0 else 3
                    for _ in range(count):
                        a=rng.uniform(0,TAU);rad=rng.uniform(.08,.37)
                        q=p+np.array([rad*np.cos(a),.7*rad*np.sin(a),0])
                        bad.add(cell(q,.074,RED,rng.uniform(-.6,.6)))
                caption='Distort population size'
            groups.append(g);failures.append(bad)
        self.play(LaggedStart(*[FadeIn(g) for g in groups],lag_ratio=.2),run_time=1.5)
        self.wait(1)
        for fail in failures:
            self.play(FadeIn(fail),run_time=1.3);self.wait(1.4)
        self.wait(4.6)


from geometry import height, solve, INITIAL, TERMINAL, STARTS, SHAPES

def primary_point(x,y):
    return np.array([-3.55+.98*x,-.27+.80*y,0])

def secondary_point(x,y,z=0):
    return np.array([3.55+.90*x+.12*y,-.33+.68*y+.58*z,0])

def visible(x,y,z):
    # Orthographic depth test against the Gaussian graph, looking from negative y.
    d=np.linspace(.045,max(.05,y+2.1),55)
    xx=x+(.12/.90)*d; yy=y-d
    ray_z=z+(.68/.58)*d
    return not np.any(height(xx,yy)>ray_z+.025)

def surface_path(points,color):
    segments=VGroup();part=[]
    for x,y in points:
        z=height(x,y)+.035
        if visible(x,y,z):part.append(secondary_point(x,y,z))
        else:
            if len(part)>1:segments.add(curve(part,color,3))
            part=[]
    if len(part)>1:segments.add(curve(part,color,3))
    return segments

def surface():
    from terrain import render_surface
    path=render_surface(Path(config.media_dir)/'surface-elevated-v2.png')
    return ImageMobject(str(path)).stretch_to_fit_width(6.5).stretch_to_fit_height(4).move_to([3.65,-.05,0])

def paths_at(c,solutions,grid):
    i=min(np.searchsorted(grid,c,side='right')-1,len(grid)-2);i=max(i,0)
    a=(c-grid[i])/(grid[i+1]-grid[i])
    return [(1-a)*s[i]+a*s[i+1] for s in solutions]

class COATIIntro(LossPanels,Scene):
    def construct(self):
        self.camera.background_color=BG
        title=words('When geometry changes the route',31).move_to([0,3.32,0])
        sub=words('Two initial populations. Five destinations.',20,MUTED).move_to([0,2.76,0])
        primary_label=words('Primary · 2D projection',18,BLUE).move_to([-3.55,2.16,0])
        secondary_label=words('Secondary · 3D geometry',18,ORANGE).move_to([3.55,2.16,0])
        self.add(title,sub)
        clouds=VGroup()
        for (x,y),(sx,sy,angle) in zip(np.vstack([INITIAL,TERMINAL]),SHAPES):
            cloud=bump(ORIGIN,sx,sy,BLUE).rotate(angle).apply_matrix(np.diag([.98,.80,1])).move_to(primary_point(x,y))
            clouds.add(cloud)
        self.play(FadeIn(primary_label),FadeIn(secondary_label),FadeIn(clouds),run_time=1.2)
        terrain=surface()
        rng=np.random.default_rng(8)
        offsets=rng.normal(size=(12,2))*[.08,.065]
        s_clouds=VGroup()
        for (x,y),(sx,sy,angle) in zip(np.vstack([INITIAL,TERMINAL]),SHAPES):
            for r in [3.1,2.0,1.0]:
                points=[]
                for a in np.linspace(0,TAU,50):
                    dx=sx*r*np.cos(a);dy=sy*r*np.sin(a)
                    xx=x+dx*np.cos(angle)-dy*np.sin(angle); yy=y+dx*np.sin(angle)+dy*np.cos(angle)
                    points.append(secondary_point(xx,yy,height(xx,yy)+.018))
                contour=VMobject(stroke_color=ORANGE,stroke_width=1.5,stroke_opacity=.78,fill_color=ORANGE,fill_opacity=.025).set_points_as_corners(points)
                contour.close_path()
                s_clouds.add(contour)
        map_eq=formula(r'T:\mathcal{X}\longrightarrow\mathcal{Y}',.44).move_to([0,-2.62,0])
        self.play(FadeIn(terrain),FadeIn(s_clouds),FadeIn(map_eq),run_time=1.6)
        self.wait(3)
        self.play(FadeOut(map_eq),run_time=.4)
        ode=formula(r'\dot{x}_t=u_\theta(x_t,t)\qquad y_t=T(x_t)',.43).move_to([0,-2.67,0])
        next_sub=words('Without sync: a straight path crosses the bump',20,MUTED).move_to(sub)
        self.play(FadeOut(sub),FadeIn(next_sub),FadeIn(ode),run_time=.7);sub=next_sub
        grid=np.linspace(0,1,17)
        solutions=[]
        for start,end in zip(STARTS,TERMINAL):
            solutions.append(np.array([solve(c,start,end)[0] for c in grid]))
        def path_mobjects(c,color=BLUE):
            pths=paths_at(c,solutions,grid)
            return VGroup(*[curve([primary_point(x,y) for x,y in p],color,3) for p in pths],*[surface_path(p,color) for p in pths])
        straight=path_mobjects(0,'#8A9BAA')
        self.play(Create(straight),run_time=1.3)
        progress=ValueTracker(0)
        def particles(c):
            arr=paths_at(c,solutions,grid);out=VGroup()
            for p in arr:
                u=progress.get_value()*(len(p)-1);i=min(int(u),len(p)-2);a=u-i;x,y=(1-a)*p[i]+a*p[i+1]
                for dx,dy in offsets[::3]:
                    out.add(Dot(primary_point(x+dx,y+dy),radius=.045,color=BLUE))
                    if visible(x+dx,y+dy,height(x+dx,y+dy)+.04):
                        out.add(Dot(secondary_point(x+dx,y+dy,height(x+dx,y+dy)+.04),radius=.045,color=ORANGE))
            return out
        dots=always_redraw(lambda:particles(0))
        self.add(dots);self.play(progress.animate.set_value(1),run_time=8,rate_func=linear);self.wait(1)
        self.play(FadeOut(dots),FadeOut(ode),run_time=.4)
        next_sub=words('With sync: secondary geometry guides the route',20,MUTED).move_to(sub)
        self.play(FadeOut(sub),FadeIn(next_sub),run_time=.6);sub=next_sub
        c=ValueTracker(0)
        routes=always_redraw(lambda:path_mobjects(c.get_value()))
        label=formula('C_y',.34,ORANGE).move_to([-2.9,-2.56,0])
        bar=Line([-2.25,-2.56,0],[1.8,-2.56,0],color=GRID,stroke_width=7)
        knob=Dot(radius=.095,color=ORANGE).add_updater(lambda m:m.move_to([-2.25+4.05*c.get_value(),-2.56,0]))
        num=always_redraw(lambda:words(f'{c.get_value():.2f}',23,ORANGE).move_to([2.5,-2.56,0]))
        obj=formula(r'\min_x\;(1-C_y)\mathcal{A}_{\mathcal{X}}[x]+C_y\mathcal{A}_{\mathcal{Y}}[T\circ x]',.39).move_to([0,-3.18,0])
        self.add(routes)
        self.play(FadeIn(label),FadeIn(bar),FadeIn(knob),FadeIn(num),FadeIn(obj),run_time=.6)
        self.play(c.animate.set_value(1),run_time=6,rate_func=linear)
        self.wait(2)
        # Make the learned vector field explicit before integrating any new path.
        for m in [routes,knob,num]:m.clear_updaters()
        self.play(*[FadeOut(m) for m in [routes,straight,label,bar,knob,num,obj]],run_time=.6)
        next_sub=words('First learn the local velocity field',20,MUTED).move_to(sub)
        self.play(FadeOut(sub),FadeIn(next_sub),run_time=.5);sub=next_sub
        from neural_field import NeuralField
        neural=NeuralField();integrated=neural.integrate()
        layers=[]
        for xx,count in [(-.65,3),(0,4),(.65,3)]:
            layers.append(VGroup(*[Circle(radius=.065,stroke_color=BLUE,stroke_width=1.4,fill_color=BG,fill_opacity=1).move_to([xx,-2.62+(j-(count-1)/2)*.20,0]) for j in range(count)]))
        links=VGroup(*[Line(a.get_center(),b.get_center(),stroke_color='#A8BCCB',stroke_width=.7) for la,lb in zip(layers[:-1],layers[1:]) for a in la for b in lb])
        network=VGroup(links,*layers)
        inp=formula('(x,t)',.30).move_to([-1.50,-2.62,0]);out=formula(r'u_\theta(x,t)',.32,BLUE).move_to([1.72,-2.62,0])
        self.play(FadeIn(network),FadeIn(inp),FadeIn(out),run_time=.8)
        self.play(Indicate(layers[0],color=BLUE),run_time=.8)
        self.play(Indicate(layers[1],color=BLUE),run_time=.8)
        self.play(Indicate(layers[2],color=BLUE),run_time=.8)
        self.wait(1.5)
        arrow_time=ValueTracker(.4)
        grid_points=np.array([(x,y) for x in np.arange(-2.5,2.65,.42) for y in np.arange(-1.8,2.15,.38)])
        def field_arrows():
            t=arrow_time.get_value();idx=min(int(t*(len(integrated)-1)),len(integrated)-1)
            centers=integrated[idx]
            active=np.min(np.linalg.norm(grid_points[:,None,:]-centers[None,:,:],axis=2),axis=1)<.77
            points=grid_points[active];vectors=neural(points,t);g=VGroup()
            for p,v in zip(points,vectors):
                mag=np.linalg.norm(v)
                if mag<.05:continue
                delta=v/mag*min(.32,.075*mag)
                a=primary_point(*p);b=primary_point(*(p+delta))
                g.add(Arrow(a,b,buff=0,color=BLUE,stroke_width=1.8,max_tip_length_to_length_ratio=.32,max_stroke_width_to_length_ratio=7).set_opacity(.65))
            return g
        arrows=always_redraw(field_arrows)
        next_sub=words('The fitted vector field tells cells where to move',20,MUTED).move_to(sub)
        self.play(FadeOut(sub),FadeIn(next_sub),FadeIn(arrows),run_time=.8);sub=next_sub
        self.wait(3)
        self.play(FadeOut(network),FadeOut(inp),FadeOut(out),run_time=.5)
        ode=formula(r'x_t=x_0+\int_0^t u_\theta(x_s,s)\,ds',.42).move_to([0,-2.55,0])
        t_label=always_redraw(lambda:words(f't = {arrow_time.get_value():.2f}',20,BLUE).move_to([0,-3.16,0]))
        next_sub=words('Neural ODE: integrate the field to generate trajectories',20,MUTED).move_to(sub)
        self.play(FadeOut(sub),FadeIn(next_sub),FadeIn(ode),run_time=.6);sub=next_sub
        arrow_time.set_value(0)
        def evolving_paths():
            u=arrow_time.get_value()*(len(integrated)-1);k=min(int(u),len(integrated)-2);a=u-k
            tip=(1-a)*integrated[k]+a*integrated[k+1]
            lines=VGroup();dots=VGroup()
            for j in range(5):
                p=np.vstack([integrated[:k+1,j],tip[j]])
                if len(p)>1 and np.linalg.norm(p[-1]-p[0])>.001:
                    lines.add(curve([primary_point(x,y) for x,y in p],BLUE,3),surface_path(p,ORANGE))
                x,y=tip[j];dots.add(cell(primary_point(x,y),.073,BLUE))
                if visible(x,y,height(x,y)+.04):dots.add(cell(secondary_point(x,y,height(x,y)+.04),.064,ORANGE))
            return VGroup(lines,dots)
        advancing=always_redraw(evolving_paths)
        self.add(advancing,t_label)
        self.play(arrow_time.animate.set_value(1),run_time=14,rate_func=linear)
        self.wait(3)
        for m in [arrows,advancing,t_label]:m.clear_updaters()
        self.play(*[FadeOut(m) for m in list(self.mobjects)],run_time=.7)
        self.ablations()
