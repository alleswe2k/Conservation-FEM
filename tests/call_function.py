from functions_eps import PDE_solver
import numpy as np

def initial_condition(x, r0=0.25, x0_1=0.3, x0_2=0):
    return 1/2*(1-np.tanh(((x[0]-x0_1)**2+(x[1]-x0_2)**2)/r0**2 - 1))


solver = PDE_solver(hmax=1/64, T=1, initial_condition=initial_condition)

solver.solve()