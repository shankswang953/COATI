"""COATI balanced-core explainer. All motion is analytic illustration, not fitted output.

Render: python -m manim -qh animations/coati_intro.py COATIIntro
Formulas are vector SVGs rendered by Matplotlib mathtext; no system TeX is required.
"""
from pathlib import Path
import hashlib
import numpy as np
from matplotlib.mathtext import math_to_image
from matplotlib import rc_context
from manim import *

BG = "#101720"
FG = "#F4F7FC"
MUTED = "#A4B3C6"
BLUE = "#48ADFF"
ORANGE = "#FFA044"
GRID = "#283647"
config.background_color = BG


def words(s, size=25, color=FG):
    m = Text(s, font="Arial", font_size=size, color=color)
    if m.width > 12.4:
        m.scale_to_fit_width(12.4)
    return m


def formula(s, height=.40, color=FG):
    folder = Path(config.media_dir) / "formula_svg"
    folder.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256((s+color+"transparent-v2").encode()).hexdigest()[:16]
    path = folder / (name + ".svg")
    if not path.exists():
        with rc_context({"savefig.transparent": True}):
            math_to_image("$"+s+"$", path, format="svg", color=color, dpi=160)
    m = SVGMobject(str(path)).set_height(height)
    if m.width > 12.5:
        m.scale_to_fit_width(12.5)
    return m


class COATIIntro(Scene):
    def construct(self):
        self.camera.background_color = BG
        title = words("From snapshots to coupled trajectories", 34).move_to([0,3.4,0])
        subtitle = words("01  /  Observe paired distributions at discrete times", 23, MUTED).move_to([0,2.87,0])
        footer = words("COATI  /  Balanced core · schematic illustration", 15, MUTED).move_to([0,-3.72,0])
        self.add(title, subtitle, footer)
        # Horizontal position is a schematic progress/time coordinate, not a fitted embedding.
        xs = [-4.7, 0, 4.7]
        centers = [1.15, -.95]
        boxes = VGroup(*[RoundedRectangle(width=12.8,height=1.78,corner_radius=.12,
                            stroke_color=GRID,stroke_width=1.3,fill_color="#141E2A",fill_opacity=1).move_to([0,y,0]) for y in centers])
        names = VGroup(words("RNA / primary",19,BLUE).move_to([-5.05,2.25,0]),
                       words("ATAC / secondary",19,ORANGE).move_to([-5.05,.14,0]))
        times = VGroup(*[words(t,18,MUTED).move_to([x,2.23,0]) for x,t in zip(xs,["t = 0","t = 1/2","t = 1"])])
        # Put time labels above the cloud regions, with modality labels on the far left.
        names[0].move_to([-5.35,1.87,0]); names[1].move_to([-5.35,-.23,0])
        self.play(FadeIn(boxes),FadeIn(names),FadeIn(times),run_time=1)
        rng=np.random.default_rng(17)
        offsets=rng.normal(0,.12,32)
        jitter=rng.normal(0,.10,32)
        branch=np.where(np.arange(32)%2,1,-1)
        def state_y(t,j):
            return offsets[j]*(1-.25*t)+branch[j]*.48*t*t + .16*np.sin(np.pi*t)
        def mapped_y(t,j):
            z=state_y(t,j)
            return 1.08*z+.12*np.sin(np.pi*t)+.08*np.sin(4*z)
        def position(t,j,secondary=False):
            return np.array([-4.7+9.4*t+jitter[j],centers[1 if secondary else 0]+(mapped_y(t,j) if secondary else state_y(t,j)),0])
        clouds=[]
        for row,color in [(False,BLUE),(True,ORANGE)]:
            for t in [0,.5,1]:
                cloud=VGroup(*[Dot(position(t,j,row),radius=.033,color=color,fill_opacity=.65) for j in range(32)])
                clouds.append(cloud)
        self.play(LaggedStart(*[FadeIn(c) for c in clouds],lag_ratio=.15),run_time=2)
        note=words("Paired within each time point. The paths between times are unobserved.",20).move_to([0,-2.20,0])
        eq=formula(r"\widehat{\rho}_i=\frac{1}{n_i}\sum_j\delta_{x_{ij}}\qquad\widehat{\xi}_i=\frac{1}{n_i}\sum_j\delta_{y_{ij}}",.49).move_to([0,-2.88,0])
        pairing=VGroup(*[DashedLine(position(.5,j),position(.5,j,True),dash_length=.07,color=MUTED,stroke_width=1,stroke_opacity=.5) for j in [3,16,25]])
        self.play(FadeIn(note),FadeIn(eq),Create(pairing),run_time=1)
        self.wait(2.5)
        self.play(FadeOut(pairing),FadeOut(note),FadeOut(eq),run_time=.5)
        # Act 2: a single primary ODE; the orange trajectory is not independently learned.
        new=words("02  /  Learn a continuous flow in the primary space",23,MUTED).move_to(subtitle)
        self.play(Transform(subtitle,new),run_time=.6)
        blue_paths=VGroup(*[ParametricFunction(lambda t,j=j:position(t,j),t_range=[0,1,.02],color=BLUE,stroke_width=1.6,stroke_opacity=.45) for j in range(0,32,3)])
        ode=formula(r"\frac{d x_t}{dt}=u_\theta(x_t,t),\qquad x_0\sim\widehat{\rho}_0",.48,BLUE).move_to([0,-2.55,0])
        note=words("A neural velocity field connects the observed distributions.",21).move_to([0,-3.22,0])
        self.play(FadeIn(ode),FadeIn(note),LaggedStart(*[Create(p) for p in blue_paths],lag_ratio=.02),run_time=2.5)
        tracker=ValueTracker(0)
        moving=VGroup(*[Dot(radius=.055,color=BLUE).add_updater(lambda m,j=j:m.move_to(position(tracker.get_value(),j))) for j in range(0,32,3)])
        self.add(moving)
        self.play(tracker.animate.set_value(1),run_time=4,rate_func=linear)
        self.wait(1)
        self.play(FadeOut(moving),FadeOut(ode),FadeOut(note),run_time=.5)
        # Act 3: exactly the same particles, projected through a fixed differentiable map.
        self.play(Transform(subtitle,words("03  /  Map the same trajectory into the second space",23,MUTED).move_to(subtitle)),run_time=.6)
        push=formula(r"y_t=T_\omega(x_t,t),\qquad \xi_t=(T_{\omega,t})_{\#}\rho_t",.43,ORANGE).move_to([0,-2.37,0])
        chain=formula(r"\dot y_t=D_xT_\omega(x_t,t)\,u_\theta(x_t,t)+\partial_tT_\omega(x_t,t)",.40).move_to([0,-3.04,0])
        orange_paths=VGroup(*[ParametricFunction(lambda t,j=j:position(t,j,True),t_range=[0,1,.02],color=ORANGE,stroke_width=1.6,stroke_opacity=.45) for j in range(0,32,3)])
        self.play(FadeIn(push),FadeIn(chain),LaggedStart(*[Create(p) for p in orange_paths],lag_ratio=.02),run_time=2)
        tracker.set_value(0)
        blue=VGroup(*[Dot(radius=.05,color=BLUE).add_updater(lambda m,j=j:m.move_to(position(tracker.get_value(),j))) for j in range(0,32,3)])
        orange=VGroup(*[Dot(radius=.05,color=ORANGE).add_updater(lambda m,j=j:m.move_to(position(tracker.get_value(),j,True))) for j in range(0,32,3)])
        connector=always_redraw(lambda:DashedLine(position(tracker.get_value(),12),position(tracker.get_value(),12,True),color=FG,stroke_width=1.7,dash_length=.08))
        tag=words("T",23).add_updater(lambda m:m.move_to((position(tracker.get_value(),12)+position(tracker.get_value(),12,True))/2+RIGHT*.24))
        self.add(blue,orange,connector,tag)
        self.play(tracker.animate.set_value(1),run_time=5,rate_func=linear)
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in [blue,orange,connector,tag,push,chain]],run_time=.5)
        # Act 4: the secondary objective differentiates through T back to theta.
        self.play(Transform(subtitle,words("04  /  Optimize one trajectory against both spaces",23,MUTED).move_to(subtitle)),run_time=.6)
        loss=formula(r"\min_\theta\;\mathcal{L}=(1-c)\mathcal{A}_X+c\mathcal{A}_Y+\lambda_X\mathcal{S}_X+\lambda_Y\mathcal{S}_Y",.43).move_to([0,-2.38,0])
        terms=formula(r"\mathcal{A}_m=\lambda_E\mathcal{E}_m+\lambda_D\mathcal{D}_m,\qquad\mathcal{E}_X=\frac{1}{2}\mathbb{E}\int\|\dot{x}_t\|^2dt,\quad\mathcal{E}_Y=\frac{1}{2}\mathbb{E}\int\|\dot{y}_t\|^2dt",.36).move_to([0,-3.06,0])
        back=CurvedArrow([5.65,-.6,0],[5.65,.78,0],angle=-PI/2,color=ORANGE,stroke_width=2.5)
        fixed=words("T fixed during trajectory training",17,MUTED).move_to([.6,.1,0])
        self.play(FadeIn(loss),FadeIn(terms),Create(back),FadeIn(fixed),run_time=1.2)
        self.play(Indicate(orange_paths,color=ORANGE,scale_factor=1.03),Indicate(blue_paths,color=BLUE,scale_factor=1.03),run_time=1.6)
        self.wait(3)
        self.play(FadeOut(back),FadeOut(fixed),FadeOut(terms),run_time=.5)
        # Act 5: vary the displayed coefficients, not fictitious re-trained trajectories.
        self.play(Transform(subtitle,words("05  /  Choose the balance of dynamical constraints",23,MUTED).move_to(subtitle)),run_time=.6)
        c=ValueTracker(.2)
        bar=Line([-2.4,.11,0],[2.4,.11,0],color=GRID,stroke_width=8)
        bluebar=always_redraw(lambda:Line([-2.4,.11,0],[-2.4+4.8*(1-c.get_value()),.11,0],color=BLUE,stroke_width=8))
        orangebar=always_redraw(lambda:Line([-2.4+4.8*(1-c.get_value()),.11,0],[2.4,.11,0],color=ORANGE,stroke_width=8))
        cx=DecimalNumber(.8,mob_class=Text,num_decimal_places=2,font_size=23,color=BLUE).move_to([-3.05,.11,0]).add_updater(lambda m:m.set_value(1-c.get_value()))
        cy=DecimalNumber(.2,mob_class=Text,num_decimal_places=2,font_size=23,color=ORANGE).move_to([3.05,.11,0]).add_updater(lambda m:m.set_value(c.get_value()))
        label=words("Primary emphasis                     Secondary emphasis",18,MUTED).move_to([0,-.23,0])
        note=words("c changes energy + manifold emphasis; both marginal constraints remain active.",20).move_to([0,-3.13,0])
        self.play(FadeIn(bar),FadeIn(bluebar),FadeIn(orangebar),FadeIn(cx),FadeIn(cy),FadeIn(label),FadeIn(note),run_time=1)
        self.play(c.animate.set_value(.8),run_time=3.5,rate_func=smooth)
        self.wait(1.5)
        self.play(c.animate.set_value(.5),run_time=2,rate_func=smooth)
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in [bar,bluebar,orangebar,cx,cy,label,note,loss]],run_time=.6)
        self.play(Transform(subtitle,words("One learned flow. Two coupled geometries.",26,FG).move_to(subtitle)),run_time=.8)
        final=formula(r"\dot{x}_t=u_\theta(x_t,t)\quad\longrightarrow\quad y_t=T_\omega(x_t,t)\quad\longrightarrow\quad\min_\theta\mathcal{L}",.43).move_to([0,-2.52,0])
        note=words("Learn in one space. Evaluate and constrain in both.",23).move_to([0,-3.19,0])
        self.play(FadeIn(final),FadeIn(note),run_time=.7)
        self.wait(3)
