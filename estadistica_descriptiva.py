import math
import support as spp
import numpy as np
from manim import *
config.max_files_cached=290
def get_csv_dots(axes:Axes, vectores,radius = 0.04,color = YELLOW):
  return [Dot(axes.coords_to_point(r[0],r[1]), radius = radius, color=color) for r in vectores]

def get_line_to_straigt(graph:'Grapher',dot,func):pass
def get_lines_to_straight(graph:'Grapher',dots:list,func):
  return [DashedLine(graph.axes.coords_to_point(*dot),graph.axes.coords_to_point(dot[0], func(dot[0]))) for dot in dots]

def get_line_x(m,b,y=0):
  return (y-b)/m

def get_rss(func,vectors):
  suma =0
  for v in vectors:
    suma+=math.pow((func(v[0])-v[1]),2)
  return suma
def sum_epsilon(func,vectors):
  suma = 0
  for v in vectors:
    suma+=v[1]-func(v[0])
  return suma
class Count(Animation):
    def __init__(self, number: DecimalNumber, start: float=None, end: float=0, **kwargs) -> None:
        # Pass number as the mobject of the animation
        super().__init__(number,  **kwargs)
        # Set start and end
        self.start = start if start!=None else number.get_value()
        self.end = end

    def interpolate_mobject(self, alpha: float) -> None:
        # Set value of DecimalNumber according to alpha
        value = self.start + (alpha * (self.end - self.start))
        if alpha>0.93: value = self.end
        self.mobject.set_value(value)

        

class Grapher():
  __M = 1/2
  __B = 1.2
  def __init__(self,xl,yl,xs,ys, scale = 1, line_func=None,line_color = RED):
    self.axes=Axes([0,xl,xs],[0,yl,ys],x_length=8,y_length=8,tips=False).add_coordinates().scale(.6*scale)
    self.labels = self.axes.get_axis_labels(x_label="x", y_label="y").set_color(YELLOW)
    self.labels[0].scale(0.6*scale)
    self.labels[1].scale(0.6*scale)
    self.xl = xl
    self.line_color=line_color
    if not line_func is None: self.line = self.graph_function(line_func,xl)
    else: self.line = None
  def graph_function(self, func, domain,color=WHITE):
    return self.axes.plot(func, domain,color=color)
  def set_line(self,func,domain):
    self.line = self.graph_function(func,domain,self.line_color)

  def transform_line(self,bf,offset,m=__M,b=__B, **kwargs):
    if not (self.line is None):
      if m is None: m = Grapher.__M
      if b is None: b = Grapher.__B
      range = [max(0,get_line_x(m,b)),min(offset,get_line_x(m,b,offset))]\
              if m>0 else\
              [max(0,get_line_x(m,b,offset)),min(8,max(0,get_line_x(m,b)))]\
              if m<0 else\
              [0,8]
      if range[0]==0 and range[1]==0: return Uncreate(self.line)
      return Transform(self.line, self.graph_function(lambda x:bf(x,m,b),range,self.line_color).set_z_index(3),**kwargs)

class equation_solver():
  def __init__(self, states:list[spp.Equation] | tuple[spp.Equation]):
    if not isinstance(states, (list,tuple)) or any(not isinstance(eq,spp.Equation) for eq in states):raise(Exception("Not valid equations!"))
    self.states:list[spp.Equation] = []
    for eq in states:
      # if not isinstance(eq_parts,list): raise(Exception("Not valid equation!"))
      self.states.append(eq)

  def to_solve(self,position=None, run_time = 1, mismathes = True):
    transforms =[]
    for i in range(len(self.states)):
      if not position is None: self.states[i].eq.move_to(position)
      if i ==0: transforms.append([Write(self.states[0].eq),self.states[0].wait])
      else:
        transforms.append([TransformMatchingTex(self.states[i-1].eq, self.states[i].eq, mismathes, run_time=run_time),self.states[i].wait])
    self.states = self.states[-1]
    return transforms, self.states.eq

  
class Presentacion(Scene):
  def construct(self):
    mxTemplate = TexTemplate()
    mxTemplate.add_to_preamble(r"\usepackage{mathrsfs}")
    DOM = 8
    vectors = np.array(spp.coords_from_csv("vectores"))
    coeficientes = np.array(spp.coords_from_csv("tabla"))
    sumatoria_coeficientes = np.array(spp.coords_from_csv("sumatorias"))
    
    base_line = lambda x,m,b:(m*x)+b
    #lambda x: base_line(x,m,b)
    title = Text("Regresión lineal")
    tabla_c = MathTable(
      coeficientes,
      col_labels=[MathTex("x_{i}").scale(1.3), MathTex("y_{i}").scale(1.3),MathTex("x_{i} \\cdot y_{i}").scale(1.3),MathTex("x_{i}^{2}").scale(1.3)],
      include_outer_lines=True).scale(0.5)
    tabla_sum = MathTable(
      sumatoria_coeficientes,
      col_labels=[MathTex("\sum\limits_{i=1}^n x_{i}").scale(1.3), MathTex("\sum\limits_{i=1}^n y_{i}").scale(1.3),MathTex("\sum\limits_{i=1}^n x_{i} \\cdot y_{i}").scale(1.3),MathTex("\sum\limits_{i=1}^n x_{i}^{2}").scale(1.3),MathTex("( \sum\limits_{i=1}^n  x_{i} )^{2}").scale(1.3)],
      include_outer_lines=True).scale(0.4).next_to(title,DOWN,0.4
    )
    name = Text("Carlos David Romero Restrepo",font_size=20).move_to(title.get_coord(2)+(DOWN*1.3))
    introduction = Text(
      "Modelo probabilístico, permite hallar la relación entre un conjunto de datos\n"
      "dependientes y uno o mas conjuntos de datos independientes",
      font_size=25
    )
    problem = Text(
      "Se recolectaron datos acerca de diferentes computadores, y se logra observar\n"
      "una relación  entre varios de los  datos. Los siguientes datos son la  cantidad\n"
      "máxima de memoria (Mb) y el rendimiento relativo de la cpu, respectivamente",
      font_size=25
    )
    
    self.play(Write(title),run_time=2)
    self.play(title.animate._set_color_by_t2c({"lineal":YELLOW}), rate_func=rush_into)
    self.play(Write(name),Indicate(title[9:],color="#ffff77"))
    self.wait()
    self.play(title.animate.next_to(introduction, UP, 0.5).scale(0.9), Unwrite(name,run_time=0.5))
    self.wait()
    self.play(Write(introduction))
    self.play(introduction.animate._set_color_by_t2c({"probabilístico":YELLOW, "dependientes":YELLOW, "independientes":YELLOW}))
    self.wait()
    self.play(title.animate.to_edge(UP,0.5).scale(.7),Uncreate(introduction))
    graph = Grapher(8,34,1,5)
    line_eq=spp.Equation("y = m x + b")
    
    self.play(Write(line_eq.eq))
    self.play(line_eq.eq.animate.scale(1.5))
    self.wait(3)
    self.play(line_eq.eq.animate.next_to(graph.axes,buff=1).scale(0.66),DrawBorderThenFill(graph.axes),Write(graph.labels))
    self.play(graph.labels.animate.set_color(WHITE), run_time=1)
    graph.set_line(lambda x:15,[0,DOM])
    dots = VGroup(*get_csv_dots(graph.axes,vectors))
    lns_v = VGroup(*get_lines_to_straight(graph,vectors, lambda x: 15))
    lns_v.set_z_index(-1)
    teaching = lns_v[4].copy()
    dot = dots[4].copy().scale(1.4)
    dot_coords = MathTex("(2,19)").next_to(dot,RIGHT,UP*0.2).scale(0.8)
    self.play(Create(dot))
    self.play(Create(graph.line))
    self.wait(2)

    self.play(Create(teaching))
    self.wait()
    self.clear()
    self.add(title,teaching)
    self.play(teaching.animate.rotate(PI/2).scale(3).center())
    epsilon = Line(teaching.get_start(),teaching.get_end(),color=RED).next_to(teaching,UP)
    error = MathTex("\\varepsilon").next_to(epsilon,UP,0.3).set_color(RED)
    self.play(Create(epsilon,run_time=1.5),Write(error,run_time=2))
    self.wait(1.5)
    complete_eq = MathTex("y = m x + b +","\\varepsilon")
    self.play(FadeOut(epsilon,teaching),error.animate.set_color(WHITE).center().scale(1.5))
    self.wait()
    self.play(TransformMatchingTex(error,complete_eq))
    del error
    self.wait()

    table = MathTable(vectors,col_labels=[MathTex("x_{i}").scale(1.3), MathTex("y_{i}").scale(1.3)],include_outer_lines=True).scale(0.4).next_to(title,DOWN,0.4)
    table.get_labels().set_color(YELLOW)
    self.play(Unwrite(complete_eq))
    self.clear()
    t_copy = title.copy()
    self.add(t_copy)
    self.play(Write(problem),t_copy.animate.next_to(problem,UP,0.5).scale(1.1))
    self.wait()
    self.play(Transform(t_copy,title))
    self.clear()
    del t_copy
    self.add(title)
    self.play(Write(table.get_labels()),DrawBorderThenFill(VGroup(table.get_vertical_lines(),table.get_horizontal_lines())))
    self.draw_vgroup_ease_out(table.get_rows()[1:],i_time=0.4)
    self.wait()
    self.play(table.animate.next_to(graph.axes,LEFT,0.7))
    self.play(Write(line_eq.eq))
    self.wait()
    self.play(Indicate(complete_eq),DrawBorderThenFill(graph.axes),Write(graph.labels),complete_eq.animate.next_to(graph.axes,buff=1))
    self.play(complete_eq.animate.next_to(graph.axes,0.5))
    self.draw_vgroup_ease_out(dots,i_time=0.4)
    func_not=MathTex("f(m,b) = m x + b").move_to(line_eq.eq).scale(0.7)
    self.play(TransformMatchingTex(line_eq.eq, func_not,True))
    replacing = MathTex("f(0,15) = 0 x + 15").move_to(line_eq.eq).scale(.7)
    self.wait()
    self.play(TransformMatchingTex(func_not,replacing,True))
    self.wait()
    self.play(Create(graph.line))
    self.wait()
    self.play(Create(lns_v))
    error_eq = [
      
      spp.Equation(complete_eq.get_tex_string()),
      spp.Equation("y_{i} = mx_{i} + b + \\varepsilon_{i}",wait=2),
      spp.Equation("\\varepsilon_{i} = y_{i} - ( mx_{i} + b )",wait=3),
      spp.Equation("\\varepsilon_{i} = y_{i} - mx_{i} - b")
    ]
    line_eq.eq.shift(LEFT*0.7)
    proccess, complete_eq = equation_solver(error_eq).to_solve(position=line_eq.eq,run_time=1.4)
    self.play(replacing.animate.shift(UP*1.5))
    self.run_with_waits(proccess)
    self.wait(3)
    summary = MathTex("\sum\limits_{i=1}^n \\varepsilon_{i}").next_to(replacing,DOWN,1.5).shift(LEFT*1.2  )
    self.play(TransformMatchingTex(complete_eq,summary,True))
    self.wait(2)
    sumatoria = sum_epsilon(lambda x:15,vectors)
    summ = MathTex(summary.get_tex_string(),f" = ").move_to(summary)
    number = DecimalNumber()
    number.next_to(summ)
    
    self.play(TransformMatchingTex(summary,summ,True),Create(number,run_time=2))
    self.wait()
    self.play(Count(number,None,sumatoria))
    sumatoria = get_rss(lambda x:15, vectors)

    summary=MathTex("\sum\limits_{i=1}^n \\varepsilon_{i} ^{2} =").move_to(summ)
    for l in dots:
      self.play(Indicate(l),run_time=0.2)
    self.play(Indicate(number))
    self.wait(0.5)
    self.play(TransformMatchingTex(summ,summary,True),number.animate.next_to(summary))
    self.play(Count(number,None,sumatoria))
    for m,b in [[0,20],[3.125,9],[-1/4,16],[2,17]]:

      func = lambda x: base_line(x,m,b)
      target = func_not.get_tex_string()
      target = target.replace('m',str(round(m,2)))
      target = target.replace('b', str(round(b,2)))
      target = MathTex(target).move_to(replacing).scale(0.7)
      sumatoria = get_rss(lambda x: base_line(x,m,b), vectors)
      self.play(
        graph.transform_line(base_line, 34,m,b),
        Transform(lns_v,VGroup(*get_lines_to_straight(graph,vectors,lambda x: base_line(x,m,b)))),
        TransformMatchingTex(replacing,target,True),
        Count(number,None,sumatoria),
        run_time=2
      )
      replacing = target
    self.wait()
    self.play(Unwrite(number))
    error_eq = [
      spp.Equation(summary),
      spp.Equation("\sum\limits_{i=1}^n \\varepsilon_{i}^{2} = \sum\limits_{i=1}^n ( y_{i} - mx_{i} - b ) ^{2}")
    ]
    proccess, line_eq.eq = equation_solver(error_eq).to_solve(run_time=1.4,position = summary)
    self.run_with_waits(proccess)
    self.clear()
    self.add(title)
    self.play(line_eq.eq.animate.center())
    derivation = MathTex("S_{r} = \sum\limits_{i=1}^n ( y_{i} - mx_{i} - b ) ^{2}").scale(0.7)
    self.play(TransformMatchingTex(line_eq.eq,derivation,True))
    derivada_m = [
      spp.Equation("\\frac{\partial S_{r}}{\partial m}"),
      spp.Equation("\\frac{\partial S_{r}}{\partial m} ; \\frac{\partial S_{r}}{\partial m}",wait=2),
      spp.Equation("\\frac{\partial S_{r}}{\partial m} = \\frac{\partial [ \sum\limits_{i=1}^n ( y_{i} - mx_{i} - b ) ^{2} ]}{\partial m}"),
      spp.Equation("\\frac{\partial S_{r}}{\partial m} = \sum\limits_{i=1}^n 2 ( y_{i} - mx_{i} - b )x_{i}"),
      spp.Equation("\\frac{\partial S_{r}}{\partial m} = 2 \sum\limits_{i=1}^n ( y_{i} - mx_{i} - b )x_{i}"),
      spp.Equation("\\frac{\partial S_{r}}{\partial m} = 2 \sum\limits_{i=1}^n ( y_{i} x_{i} - mx_{i}^{2} - bx_{i} )")
    ]
    derivada_b = [
      spp.Equation("\\frac{\partial S_{r}}{\partial m}",wait=2),
      spp.Equation("\\frac{\partial S_{r}}{\partial b} = \\frac{\partial [ \sum\limits_{i=1}^n ( y_{i} - mx_{i} - b ) ^{2} ]}{\partial b}"),
      spp.Equation("\\frac{\partial S_{r}}{\partial b} = \sum\limits_{i=1}^n 2 ( y_{i} - mx_{i} - b )"),
      spp.Equation("\\frac{\partial S_{r}}{\partial b} = 2 \sum\limits_{i=1}^n ( y_{i} - mx_{i} - b )"),
    ]

    # spp.Equation("2  \sum\limits_{i=1}^n ( y_{i} - mx_{i} - b )x_{i} = 0"),
    # spp.Equation("\sum\limits_{i=1}^n ( y_{i} - mx_{i} - b )x_{i} = 0"),
    self.play(derivation.animate.next_to(title,DOWN,0.5))
    proccess, derivada_m = equation_solver(derivada_m).to_solve(run_time=1.4)
    self.run_with_waits(proccess)
    proccess, derivada_b = equation_solver(derivada_b).to_solve(run_time=1.4, position=derivada_m.get_coord(2)+DOWN*2)
    self.run_with_waits(proccess)
    self.wait()
    mto0=MathTex("2 \sum\limits_{i=1}^n ( y_{i} x_{i} - m x_{i}^{2} - b x_{i} ) = 0")
    bto0=MathTex("2 \sum\limits_{i=1}^n ( y_{i} - m x_{i} - b ) = 0")

    derivada_m = [
      spp.Equation(mto0.get_tex_string()),
      spp.Equation("- m \sum\limits_{i=1}^n x_{i}^{2} - b \sum\limits_{i=1}^n x_{i} = - \sum\limits_{i=1}^n y_{i} x_{i}"),
      spp.Equation("m \sum\limits_{i=1}^n x_{i}^{2} + b \sum\limits_{i=1}^n x_{i} = \sum\limits_{i=1}^n y_{i} x_{i}")
    ]
    derivada_b = [
      spp.Equation(bto0.get_tex_string()),
      spp.Equation("- m \sum\limits_{i=1}^n x_{i} - \sum\limits_{i=1}^n b = - \sum\limits_{i=1}^n y_{i}"),
      spp.Equation("m \sum\limits_{i=1}^n x_{i} + n b = \sum\limits_{i=1}^n y_{i}")
    ]
    self.clear()
    self.add(title,derivation)
    proccess, derivada_m = equation_solver(derivada_m).to_solve(run_time=1.4)
    self.run_with_waits(proccess)
    proccess, derivada_b = equation_solver(derivada_b).to_solve(run_time=1.4, position=derivada_m.get_coord(2)+DOWN*2.5)
    self.run_with_waits(proccess)
    self.wait()
    to_transform = VGroup(derivada_m,derivada_b)
    matrix_eq = MathTex(
      r"\begin{pmatrix} "
      r"\sum\limits_{i=1}^n x_{i}^{2} & \sum\limits_{i=1}^n x_{i}\\"
      r"\sum\limits_{i=1}^n x_{i} & n "
      r"\end{pmatrix}"
      r" · \begin{pmatrix} m \\ b \end{pmatrix}"
      r" = \begin{pmatrix} \sum\limits_{i=1}^n y_{i} x_{i} \\ \sum\limits_{i=1}^n y_{i} \end{pmatrix}"
      ,tex_template=mxTemplate
    ).next_to(derivation,DOWN,0.5)
    self.clear()
    self.add(derivation,title)
    self.play(Transform(to_transform,matrix_eq))
    m_eq = MathTex(
      "m = \\frac{n \sum\limits_{i=1}^n y_{i} x_{i} - \sum\limits_{i=1}^n x_{i} \sum\limits_{i=1}^n y_{i}}{n \sum\limits_{i=1}^n x_{i}^{2} - ( \sum\limits_{i=1}^n x_{i} )^{2}}"
    )
    b_eq = [
      spp.Equation("b = \\frac{\sum\limits_{i=1}^n y_{i} - m \sum\limits_{i=1}^n x_{i}}{n}"),
      spp.Equation("b = \\overline{y} - m \\overline{x}",wait=3)

    ]
    self.play(Transform(to_transform,m_eq))
    
    proccess, b_eq = equation_solver(b_eq).to_solve(run_time=1.4,position = m_eq.get_coord(2)+(DOWN*2.4))

    self.run_with_waits(proccess)
    self.wait()
    self.clear()
    self.add(b_eq,m_eq,title)
    self.play(b_eq.animate.scale(0.7),m_eq.animate.scale(0.7))
    self.play(Write(table.get_labels()),DrawBorderThenFill(VGroup(table.get_vertical_lines(),table.get_horizontal_lines())))
    self.draw_vgroup_ease_out(table.get_rows()[1:],i_time=0.4)
    self.wait()
    self.clear()
    self.add(table,title)
    
    m_equation = [
      spp.Equation(m_eq.get_tex_string(),wait=2),
      spp.Equation("m = \\frac{10 \\cdot ( 777.964 ) - ( 219 ) ( 29.344 ) }{ 10 \\cdot ( 141.35 ) - ( 861.07 )}",wait=2),
      spp.Equation("m = 2.449",wait=2)
    ]
    b_equation = [
      spp.Equation(b_eq.get_tex_string(),wait=2),
      spp.Equation("b = 21.9 - 2.449 \\cdot 2.934",wait=2),
      spp.Equation("b = 14.712",wait=2)
    ]
    tabla_c.next_to(title,DOWN,0.2)
    tabla_sum.next_to(title,DOWN,0.2)
    self.play(Transform(table,tabla_c))
    self.wait(3)
    self.clear()
    self.add(title)
    self.play(Transform(tabla_c,tabla_sum))
    self.wait(1.4)
    self.play(Write(MathTex("n = 10")))

    self.wait()
    self.clear()
    self.add(title)
    proccess, m_equation = equation_solver(m_equation).to_solve(run_time=1.4)
    self.run_with_waits(proccess)
    self.wait()
    self.play(FadeOut(m_equation))
    proccess, b_equation = equation_solver(b_equation).to_solve(run_time=1.4)
    self.run_with_waits(proccess)
    self.wait()
    self.play(FadeOut(b_equation))
    self.wait()
    graph.axes.to_edge(LEFT,1.2)
    graph.labels = graph.axes.get_axis_labels()
    self.play(DrawBorderThenFill(graph.axes),Write(graph.labels))
    self.play(graph.labels.animate.set_color(WHITE), run_time=1)
    graph.set_line(lambda x: base_line(x,2.45,14.7),[0,DOM])
    replacing = MathTex("f(2.45,14.7) = 2.45 x + 14.7").move_to(func_not).scale(.7).shift(UP*1.6 + LEFT*1.6)
    self.play(TransformMatchingTex(func_not, replacing,True))
    dots = VGroup(*get_csv_dots(graph.axes,vectors))
    self.play(Create(dots),Create(graph.line))
    lns_v = VGroup(*get_lines_to_straight(graph,vectors, lambda x: base_line(x,2.45,14.7)))
    lns_v.set_z_index(-1)
    self.play(Create(lns_v))
    sumatoria = get_rss(lambda x:base_line(x,2.45,14.7), vectors)
    # summary=MathTex("\sum\limits_{i=1}^n \\varepsilon_{i} ^{2} =").move_to(summ)
    summary.shift(LEFT*1.8)
    number.scale(2.3)
    number.next_to(summary)
    self.play(TransformMatchingTex(summ,summary,True),Create(number,run_time=2))
    for l in lns_v:
      self.play(Indicate(l),run_time=0.5)
    self.play(Count(number, 0,sumatoria))
    self.wait()
    self.play(Indicate(number),run_time=2)

  def draw_vgroup_ease_out(self, mobject:VGroup, i_time):
    c = len(mobject)
    for i in range(c):
      self.play(FadeIn(mobject[i]),mobject[i].animate.set_color(WHITE),run_time=i_time-((i_time/c)*i))
    
  def run_with_waits(self, animations):
    for ste in range(len(animations)):
      self.wait(animations[ste][1])
      self.play(animations[ste][0])

class Cierre(Scene):
    def construct(self):
      
      #lambda x: base_line(x,m,b)
      title = Text("Gracias por su atención")
      self.play(Write(title),run_time=1)
      self.play(Indicate(title))
      self.play(title.animate.scale(1.6))