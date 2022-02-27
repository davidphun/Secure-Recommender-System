from gurobipy import *

# Data
p = 20
q = 1961
k = 8
z = 210

a = (1 / q) + (k / z)
b = (1 / p) - (k / z)

H = a / b

privacy_budget = 1

print(H)

t = 1e-2

# Model

m = Model('p5_p6_optimization')

# Variables
p5 = m.addVar()
p6 = m.addVar()

# Objective function

m.setObjective(p5, GRB.MAXIMIZE)

# Constraints

m.addConstr(p5 <= 1.0)
m.addConstr(p5 >= 0.0)
m.addConstr(p6 <= 1.0)
m.addConstr(p6 >= 0.0)
m.addConstr(p6 == (t / a) + ((1/H) * p5))
m.addConstr(p6 >= privacy_budget * p5)

# Optimize
m.optimize()

print('p5 {}, p6 {}'.format(round(p5.x, 2), round(p6.x, 2)))

privacy_budget = max(
    p5.x / p6.x,
    p6.x / p5.x,
    (1 - p5.x) / (1 - p6.x),
    (1 - p6.x) / (1 - p5.x)
)

print(privacy_budget)