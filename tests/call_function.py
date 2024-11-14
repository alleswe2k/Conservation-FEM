from functions_eps_copy import PDE_solver
import numpy as np

def initial_condition(x, r0=0.25, x0_1=0.3, x0_2=0):
    return 1/2*(1-np.tanh(((x[0]-x0_1)**2+(x[1]-x0_2)**2)/r0**2 - 1))

def velocity_field(x):
    return np.array([-2*np.pi*x[1], 2*np.pi*x[0]])

solver = PDE_solver(initial_condition=initial_condition, f=velocity_field)

# convergence = solver.calc_convergence()
# print(convergence)

solver.solve(1/16)
solver.plot()
