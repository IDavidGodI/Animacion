from random import randint as __rng
from manim import MathTex
def __funcion(x,m=1/2,b=1.2, decm=3):
  y = (m*x) + b
  eps = __rng(8,90) if __rng(0,100)>55 else __rng(110,120)
  if __rng(0,1)==1: eps=-eps
  return (x,round(y+(eps/100),decm))

def __write_csv(file_name, matrix):
  import csv
  coords = []
  with open(f'{file_name}.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    for row in matrix:
      writer.writerow(row)
  csvFile.close()

def coords_from_csv(file_name):
  import csv
  coords = []
  with open(f'{file_name}.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
      
      coord = [float(value) for value in row]
      coords.append(coord)
  csvFile.close()
  return coords

def generate_vectors(data, x_length):
  step = x_length/data
  matrix = []
  for _ in range(0,data):
    i = step*(_+1)
    delt = __rng(-49,49)/100

    x = i-delt
    matrix.append(__funcion(x))
  __write_csv("vectores",matrix)
  return matrix

class Equation():
  def __init__(self,parts,move_to=None, wait=0, **kwargs ):
    self.wait = wait
    if isinstance(parts,MathTex): self.eq=parts
    else: self.eq = MathTex(parts,**kwargs)
    if not move_to is None: self.eq.move_to(move_to)
  
  def get_string(self):
    return self.eq.get_tex_string()



  
