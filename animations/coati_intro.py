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
    return Text(s, font='Arial', font_size=size, color=color)

def formula(s, height=.38, color=FG):
    folder = Path(config.media_dir) / 'formula_svg'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (hashlib.sha256((s+color+'transparent-v2').encode()).hexdigest()[:16]+'.svg')
    if not path.exists():
        with rc_context({'savefig.transparent': True}):
            math_to_image('$'+s+'$', path, format='svg', color=color, dpi=160)
    obj = SVGMobject(str(path)).set_height(height)
    if obj.width > 12.5: obj.scale_to_fit_width(12.5)
    return obj

def curve(points, color=BLUE, width=4):
    return VMobject(stroke_color=color, stroke_width=width).set_points_smoothly(np.array(points))

def bump(center, sx=.36, sy=.24, color=BLUE):
    # Nested level sets of a Gaussian; the center stays visible without a point cloud.
    return VGroup(*[Ellipse(width=2*sx*r,height=2*sy*r,stroke_color=color,
        stroke_width=0,stroke_opacity=0,fill_color=color,fill_opacity=.075).move_to(center)
        for r in np.linspace(2.7,.25,15)])

class LossPanels:
    def ablations(self):
        title=words('What each constraint prevents',37).move_to([0,3.25,0])
        sub=formula(r'\mathcal{L}=C_x\mathcal{A}_{\mathcal{X}}+C_y\mathcal{A}_{\mathcal{Y}}+\mathcal{L}_{\mathrm{Ref}}+\mathcal{L}_{\mathrm{Manifold}}+\mathcal{L}_{\mathrm{Mass}}',.34).move_to([0,2.7,0])
        self.play(FadeIn(title),FadeIn(sub),run_time=.6)
        centers=[-4.55,0,4.55]
        groups=[]; failures=[]
        for i,x in enumerate(centers):
            card=RoundedRectangle(width=4.22,height=4.65,corner_radius=.16,stroke_color=GRID,fill_color=WHITE,fill_opacity=1).move_to([x,-.08,0])
            eq=formula([r'\mathrm{w/o}\;\mathcal{L}_{\mathrm{Ref}}',r'\mathrm{w/o}\;\mathcal{L}_{\mathrm{Manifold}}',r'\mathrm{w/o}\;\mathcal{L}_{\mathrm{Mass}}'][i],.34).move_to([x,1.75,0])
            start=np.array([x-1.45,-.3,0]); end=np.array([x+1.35,-.15,0])
            mid=np.array([x,.65 if i==1 else .15,0])
            bg=VGroup(*[bump(p,.20,.16,'#697D91') for p in [start,mid,end]])
            if i==1:
                ridge=curve([start,mid,end], '#BAC7D1',23).set_stroke(opacity=.35)
                bg.add_to_back(ridge)
            path=curve([start,mid,end],GREEN,3)
            rings=VGroup(*[Circle(radius=.115,color=GREEN,stroke_width=2).move_to(p) for p in [start,mid,end]])
            g=VGroup(card,eq,bg,path,rings)
            if i==0:
                bad=curve([start,[x,-.8,0],[x+1.4,-.75,0]],RED,3)
                marks=VGroup(*[Circle(radius=.14,color=RED,stroke_width=2).move_to(p) for p in [[x,-.8,0],[x+1.4,-.75,0]]])
                caption='Miss the observed distributions'
            elif i==1:
                bad=curve([start,[x,-.45,0],end],RED,3)
                marks=Circle(radius=.14,color=RED,stroke_width=2).move_to([x,-.45,0])
                caption='Take a low-density shortcut'
            else:
                bad=curve([start,mid,end],RED,2)
                marks=VGroup(Circle(radius=.42,color=RED,stroke_width=2).move_to(mid),Circle(radius=.06,color=RED,stroke_width=2).move_to(end))
                caption='Get population mass wrong'
                g.add(words('Circle area represents mass',15,MUTED).move_to([x,-1.12,0]))
            label=words(caption,19).move_to([x,-1.62,0])
            if label.width>3.8:label.scale_to_fit_width(3.8)
            g.add(label)
            groups.append(g); failures.append(VGroup(bad,marks))
        self.play(LaggedStart(*[FadeIn(g) for g in groups],lag_ratio=.2),run_time=1.5)
        self.wait(1)
        for fail in failures:
            self.play(Create(fail),run_time=1.3)
            self.wait(1.4)
        legend=VGroup(words('Observed density',18,'#697D91'),words('Constrained path',18,GREEN),words('Possible failure',18,RED)).arrange(RIGHT,buff=.55).move_to([0,-2.83,0])
        note=words('Mass matching applies to the growth / unbalanced extension.',18,MUTED).move_to([0,-3.35,0])
        self.play(FadeIn(legend),FadeIn(note),run_time=.6)
        self.wait(4)

from geometry import height, solve, INITIAL, TERMINAL, STARTS

def primary_point(x,y):
    return np.array([-3.55+.98*x,-.27+.80*y,0])

def secondary_point(x,y,z=0):
    return np.array([3.55+.91*x+.29*y,-.45+.37*y+.82*z,0])

def visible(x,y,z):
    # Orthographic depth test against the Gaussian graph, looking from negative y.
    d=np.linspace(.045,max(.05,y+2.1),55)
    xx=x+(.29/.91)*d; yy=y-d
    ray_z=z+(.37/.82)*d
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
    mesh=VGroup()
    xs=np.linspace(-2.9,2.9,66);ys=np.linspace(-2.5,2.5,50)
    for j in range(len(ys)-2,-1,-1):
        for i in range(len(xs)-1):
            corners=[(xs[i],ys[j]),(xs[i+1],ys[j]),(xs[i+1],ys[j+1]),(xs[i],ys[j+1])]
            z=np.mean([height(x,y) for x,y in corners])
            col=interpolate_color(ManimColor('#FFF4E8'),ManimColor(ORANGE),min(.90,z/3))
            mesh.add(Polygon(*[secondary_point(x,y,height(x,y)) for x,y in corners],stroke_color=col,stroke_width=.18,stroke_opacity=min(1,z/.16),fill_color=col,fill_opacity=min(1,z/.16)))
    return mesh

def paths_at(c,solutions,grid):
    i=min(np.searchsorted(grid,c,side='right')-1,len(grid)-2);i=max(i,0)
    a=(c-grid[i])/(grid[i+1]-grid[i])
    return [(1-a)*s[i]+a*s[i+1] for s in solutions]

class COATIIntro(LossPanels,Scene):
    def construct(self):
        self.camera.background_color=BG
        title=words('When geometry changes the route',38).move_to([0,3.35,0])
        sub=words('Two initial populations. Five destinations.',23,MUTED).move_to([0,2.79,0])
        primary_label=words('PRIMARY  /  2D projection',20,BLUE).move_to([-3.55,2.16,0])
        secondary_label=words('SECONDARY  /  3D geometry',20,ORANGE).move_to([3.55,2.16,0])
        footer=words('COATI · geometric toy illustration',16,MUTED).move_to([0,-3.63,0])
        self.add(title,sub,footer)
        clouds=VGroup(*[bump(primary_point(x,y),.17,.14,BLUE) for x,y in np.vstack([INITIAL,TERMINAL])])
        timelabels=VGroup(words('2 initial groups',18,MUTED).move_to([-5.1,-2.03,0]),words('5 terminal groups',18,MUTED).move_to([-1.7,1.81,0]))
        self.play(FadeIn(primary_label),FadeIn(secondary_label),FadeIn(clouds),FadeIn(timelabels),run_time=1.2)
        terrain=surface()
        rng=np.random.default_rng(8)
        offsets=rng.normal(size=(12,2))*[.08,.065]
        s_clouds=VGroup()
        for x,y in np.vstack([INITIAL,TERMINAL]):
            for r in np.linspace(2.7,.25,18):
                points=[]
                for a in np.linspace(0,TAU,50):
                    xx=x+.17*r*np.cos(a); yy=y+.17*r*np.sin(a)
                    points.append(secondary_point(xx,yy,height(xx,yy)+.018))
                contour=VMobject(stroke_width=0,fill_color=ORANGE,fill_opacity=.075).set_points_as_corners(points)
                contour.close_path()
                s_clouds.add(contour)
        map_eq=formula(r'T(x_1,x_2)=(x_1,x_2,h(x_1,x_2)),\qquad\pi(T(x))=x',.38).move_to([0,-2.62,0])
        gaussian=formula(r'h(x)=\sum_k a_k\exp\!\left(-\frac{\|x-m_k\|^2}{2\sigma_k^2}\right)',.40,ORANGE).move_to([0,-3.18,0])
        self.play(FadeIn(terrain),FadeIn(s_clouds),FadeIn(map_eq),FadeIn(gaussian),run_time=1.6)
        self.wait(3)
        self.play(FadeOut(gaussian),FadeOut(map_eq),run_time=.4)
        ode=formula(r'\dot{x}_t=u_\theta(x_t,t)\qquad y_t=T(x_t)',.43).move_to([0,-2.67,0])
        self.play(Transform(sub,words('No sync  /  Straight in projection, costly over the bump',23,MUTED).move_to(sub)),FadeIn(ode),run_time=.7)
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
        self.add(dots);self.play(progress.animate.set_value(1),run_time=4,rate_func=linear);self.wait(1)
        self.play(FadeOut(dots),FadeOut(ode),run_time=.4)
        self.play(Transform(sub,words('With sync  /  Let the secondary geometry guide the route',23,MUTED).move_to(sub)),run_time=.6)
        c=ValueTracker(0)
        routes=always_redraw(lambda:path_mobjects(c.get_value()))
        label=formula('C_y',.34,ORANGE).move_to([-2.9,-2.56,0])
        bar=Line([-2.25,-2.56,0],[1.8,-2.56,0],color=GRID,stroke_width=7)
        knob=Dot(radius=.095,color=ORANGE).add_updater(lambda m:m.move_to([-2.25+4.05*c.get_value(),-2.56,0]))
        num=DecimalNumber(0,mob_class=Text,num_decimal_places=2,font_size=24,color=ORANGE).move_to([2.5,-2.56,0]).add_updater(lambda m:m.set_value(c.get_value()))
        obj=formula(r'\min_x\;(1-C_y)\mathcal{A}_{\mathcal{X}}[x]+C_y\mathcal{A}_{\mathcal{Y}}[T\circ x]',.39).move_to([0,-3.18,0])
        legend=words('Gray: no sync     Blue: sync',17,MUTED).move_to([3.5,-2.0,0])
        self.add(routes,legend)
        self.play(FadeIn(label),FadeIn(bar),FadeIn(knob),FadeIn(num),FadeIn(obj),run_time=.6)
        self.play(c.animate.set_value(1),run_time=6,rate_func=linear)
        self.wait(2)
        progress.set_value(0);dots=always_redraw(lambda:particles(1));self.add(dots)
        self.play(progress.animate.set_value(1),run_time=4,rate_func=linear)
        self.wait(1)
        for m in [routes,knob,num,dots]:m.clear_updaters()
        self.play(*[FadeOut(m) for m in list(self.mobjects)],run_time=.7)
        self.ablations()
