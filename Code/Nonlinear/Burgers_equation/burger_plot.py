import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

import gmsh
from dolfinx.io import gmshio

from tqdm import tqdm

from dolfinx import fem, mesh, io, plot, nls, log
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from PDE_solver import PDE_solver


pde = PDE_solver()

size = 100

domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [size, size], cell_type=mesh.CellType.triangle)

V = fem.functionspace(domain, ("Lagrange", 1))
DG0 = fem.functionspace(domain, ("DG", 0))
DG1 = fem.functionspace(domain, ("DG", 1))


def velocity_field(u):
    # Apply nonlinear operators correctly to the scalar function u
    return ufl.as_vector([u,u])

def exact_solution(x): 
    t = 0.5

    u = np.zeros_like(x[0])  # Initialize the solution array with zeros
    
    # First condition
    mask1 = x[0] <= (1/2 - 3 * t / 5)
    u = np.where(mask1 & (x[1] > (1/2 + 3 * t / 20)), -0.2, u)
    u = np.where(mask1 & (x[1] <= (1/2 + 3 * t / 20)), 0.5, u)

    # Second condition
    # mask2 = (1/2 - t / 4 <= x[0]) & (x[0] <= (1/2 + t / 2))
    mask2 = ((1/2 - 3*t/5) <= x[0]) & (x[0] <= (1/2 - t/4))
    u = np.where(mask2 & (x[1] > (-8 * x[0] / 7 + 15 / 14 - 15 * t / 28)), -1, u)
    u = np.where(mask2 & (x[1] <= (-8 * x[0] / 7 + 15 / 14 - 15 * t / 28)), 0.5, u)

    # Third condition
    mask3 = (1/2 - t / 4 <= x[0]) & (x[0] <= (1/2 + t / 2))
    u = np.where(mask3 & (x[1] > (x[0] / 6 + 5 / 12 - 5 * t / 24)), -1, u)
    u = np.where(mask3 & (x[1] <= (x[0] / 6 + 5 / 12 - 5 * t / 24)), 0.5, u)

    # Fourth condition
    mask4 = (1/2 + t / 2 <= x[0]) & (x[0] <= (1/2 + 4 * t / 5))
    u = np.where(mask4 & (x[1] > (x[0] - 5 / (18 * t) * (x[0] + t - 1/2)**2)), -1, u)
    u = np.where(mask4 & (x[1] <= (x[0] - 5 / (18 * t) * (x[0] + t - 1/2)**2)), (2 * x[0] - 1) / (2 * t), u)

    # Fifth condition
    mask5 = x[0] >= (1/2 + 4 * t / 5)
    u = np.where(mask5 & (x[1] > (1/2 - t / 10)), -1, u)
    u = np.where(mask5 & (x[1] <= (1/2 - t / 10)), 0.8, u)

    return u



def initial_condition(x):
    x0, x1 = x[0], x[1]  # Extract x0 and x1 from the input array

    # Initialize the solution array with zeros
    u = np.zeros_like(x0)
    # Apply the conditions
    u = np.where((x0 <= 0.5) & (x1 >= 0.5), -0.2, u)
    u = np.where((x0 > 0.5) & (x1 >= 0.5), -1.0, u)
    u = np.where((x0 <= 0.5) & (x1 < 0.5), 0.5, u)
    u = np.where((x0 > 0.5) & (x1 < 0.5), 0.8, u)
    return u

u_exact = pde.create_vector(V, 'u_exact', exact_solution)
u_n = pde.create_vector(V, 'u_n', initial_condition)
u_old = pde.create_vector(V, 'u_old', initial_condition)



CFL = 0.2
t = 0  # Start time
T = 0.5 # Final time
dt = 0.01
num_steps = int(np.ceil(T/dt))
Cvel = 0.25
CRV = 4.0


# # Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
# # bc = fem.dirichletbc(PETSc.ScalarType(np.pi/4), fem.locate_dofs_topological(V, fdim, boundary_facets), V)
# bc = fem.dirichletbc(u_exact_boundary, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Locate boundary degrees of freedom
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

# Create a function to interpolate the exact solution
u_exact_boundary = fem.Function(V)
u_exact_boundary.interpolate(exact_solution)

# Apply the interpolated exact solution on the boundary
bc = fem.dirichletbc(u_exact_boundary, boundary_dofs)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = pde.create_vector(V, 'uh', initial_condition)
f_rh = lambda x: np.full(x.shape[1], 0, dtype = np.float64)
RH = pde.create_vector(V, 'RH', f_rh)

# Variational problem and solver
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
F = (uh*v *ufl.dx -
     u_n*v *ufl.dx + 
     0.5*dt*ufl.dot(velocity_field(uh), ufl.grad(uh))*v*ufl.dx + 
     0.5*dt*ufl.dot(velocity_field(u_n), ufl.grad(u_n))*v*ufl.dx)


nonlin_problem = NonlinearProblem(F, uh, bcs = [bc])
nonlin_solver = NewtonSolver(MPI.COMM_WORLD, nonlin_problem)
nonlin_solver.max_it = 100  # Increase maximum number of iterations
nonlin_solver.rtol = 1e-1
nonlin_solver.report = True    

h_DG = fem.Function(DG0)  # Cell-based function for hk values

cell_to_vertex_map = domain.topology.connectivity(domain.topology.dim, 0)
vertex_coords = domain.geometry.x

num_cells = domain.topology.index_map(domain.topology.dim).size_local
hk_values = np.zeros(num_cells)

for cell in range(num_cells):
    # Get the vertices of the current cell
    cell_vertices = cell_to_vertex_map.links(cell)
    coords = vertex_coords[cell_vertices]  # Coordinates of the vertices
    
    edges = [np.linalg.norm(coords[i] - coords[j]) for i in range(len(coords)) for j in range(i + 1, len(coords))]
    hk_values[cell] = min(edges) 

h_DG.x.array[:] = hk_values
v = ufl.TestFunction(V)
h_trial = ufl.TrialFunction(V)
a_h = h_trial * v * ufl.dx
L_h = h_DG * v * ufl.dx

# Solve linear system
lin_problem = LinearProblem(a_h, L_h, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
h_CG = lin_problem.solve() # returns dolfinx.fem.Function


#Take GFEM STEP
t += dt
n, converged = nonlin_solver.solve(uh)
assert (converged)
# uh.x.scatter_forward()

# Update solution at previous time step (u_n)
u_n.x.array[:] = uh.x.array
# Write solution to file

for i in tqdm(range(num_steps -1)):
    t += dt

    
    # a_R = u*v*ufl.dx
    # L_R = 1/dt*u_n * v * ufl.dx - 1/dt* u_old * v *ufl.dx + ufl.dot(velocity_field(u_n),ufl.grad(u_n))* v * ufl.dx
    # F_R = (a_R - L_R)


    F_R = (RH*v*ufl.dx - 1/dt*u_n*v *ufl.dx - 1/dt*u_old*v*ufl.dx +
        0.5*ufl.dot(velocity_field(u_n), ufl.grad(u_n))*v*ufl.dx + 
        0.5*ufl.dot(velocity_field(u_old), ufl.grad(u_old))*v*ufl.dx)
    R_problem = NonlinearProblem(F_R, RH, bcs = [bc])
    # Rh = R_problem.solve()

    Rh_problem = NewtonSolver(MPI.COMM_WORLD, R_problem)
    Rh_problem.convergence_criterion = "incremental"
    Rh_problem.max_it = 100  # Increase maximum number of iterations
    Rh_problem.rtol =  1e-1
    Rh_problem.report = True

    n, converged = Rh_problem.solve(RH)
    # n, converged = Rh.solve(uh)
    # assert (converged)
    RH.x.array[:] = RH.x.array / np.max(u_n.x.array - np.mean(u_n.x.array))
    epsilon = fem.Function(V)

    for node in range(RH.x.array.size):
        hi = h_CG.x.array[node]
        Ri = RH.x.array[node]
        w = uh.x.array[node]
        w = velocity_field(uh.x.array[node])
        fi = np.array(w, dtype = 'float')
        fi_norm = np.linalg.norm(fi)
        epsilon.x.array[node] = min(Cvel * hi * fi_norm, CRV * hi ** 2 * np.abs(Ri))
    
    F = (uh*v *ufl.dx - u_n*v *ufl.dx + 
        0.5*dt*ufl.dot(velocity_field(uh), ufl.grad(uh))*v*ufl.dx + 
        0.5*dt*ufl.dot(velocity_field(u_n), ufl.grad(u_n))*v*ufl.dx + 
        0.5*dt*epsilon*ufl.dot(ufl.grad(uh), ufl.grad(v))*ufl.dx +
        0.5*dt*epsilon*ufl.dot(ufl.grad(u_n), ufl.grad(v))*ufl.dx)
    
    problem = NonlinearProblem(F, uh, bcs = [bc])
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.max_it = 100  # Increase maximum number of iterations
    solver.rtol =  1e-1
    solver.report = True

    # Solve linear problem
    n, converged = solver.solve(uh)
    assert (converged)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_old.x.array[:] = u_n.x.array
    u_n.x.array[:] = uh.x.array


# pde.plot_solution(domain, size, RH, 'Rh', 'rv')
pde.plot_2d(domain, size, epsilon, 'Epsilon', 'epsilon')
pde.plot_2d(domain, size, RH, 'Rh', 'rh')
pde.plot_2d(domain, size, uh, 'Uh', 'uh')

  


