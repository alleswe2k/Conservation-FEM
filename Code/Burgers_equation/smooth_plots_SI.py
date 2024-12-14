import matplotlib as mpl
import pyvista
import ufl
import numpy as np
from tqdm import tqdm

from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_matrix, NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver

from Utils.helpers import get_nodal_h, smooth_vector
from Utils.SI import SI
from Utils.PDE_plot import PDE_plot

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
location_figures = os.path.join(script_dir, 'Figures/SI') # location = './Figures'
location_data = os.path.join(script_dir, 'Data/SI') # location = './Data'

pde = PDE_plot()
PLOT = False

N = 100
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [N, N], cell_type=mesh.CellType.triangle)

V = fem.functionspace(domain, ("Lagrange", 1))
DG0 = fem.functionspace(domain, ("DG", 0))

def velocity_field(u):
    # Apply nonlinear operators correctly to the scalar function u
    return ufl.as_vector([u,u])

def exact_solution(x, t=0.5):

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

l_values = [10, 6, 4, 2]

for l in l_values:
    u_exact = fem.Function(V)
    u_exact.name = "U Exact"
    u_exact.interpolate(exact_solution)

    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_n.interpolate(initial_condition)

    u_old = fem.Function(V)
    u_old.name = "u_old"
    u_old.interpolate(initial_condition)

    plot_func = fem.Function(V)
    plot_func.name = "plot_func"

    h_CG = get_nodal_h(domain)

    CFL = 0.5
    t = 0  # Start time
    T = 0.5 # Final time
    dt = CFL * min(h_CG.x.array)
    num_steps = int(np.ceil(T/dt))
    Cm = 0.5
    eps = 1e-8

    print("dt:", dt)

    si = SI(Cm, domain, eps)

    """ Creat patch dictionary """
    node_patches = si.get_patch_dictionary()


    # # Create boundary condition
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    # # bc = fem.dirichletbc(PETSc.ScalarType(np.pi/4), fem.locate_dofs_topological(V, fdim, boundary_facets), V)
    # bc = fem.dirichletbc(u_exact_boundary, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

    # Locate boundary degrees of freedom
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # Define solution variable, and interpolate initial solution for visualization in Paraview
    uh = fem.Function(V)
    uh.name = "uh"
    uh.interpolate(initial_condition)

    # Variational problem and solver
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    # Time-dependent output
    xdmf_sol = io.XDMFFile(domain.comm, f"{location_data}/sol_smooth_N{N}_l{l}.xdmf", "w")
    xdmf_sol.write_mesh(domain)

    xdmf_sol.write_function(u_n, t)

    for i in tqdm(range(num_steps)):
        t += dt
        # Create a function to interpolate the exact solution
        u_exact_boundary = fem.Function(V)
        u_exact_boundary.interpolate(lambda x: exact_solution(x, t))

        # Apply the interpolated exact solution on the boundary
        bc = fem.dirichletbc(u_exact_boundary, boundary_dofs)

        """ Assemble stiffness matrix, obtain element values """
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        A = assemble_matrix(fem.form(a), bcs=[bc])
        A.assemble()

        epsilon = si.get_epsilon_nonlinear(velocity_field, node_patches, h_CG, u_n, A, plot_func)
        
        F = (uh*v *ufl.dx - u_n*v *ufl.dx + 
            0.5*dt*ufl.dot(velocity_field(uh), ufl.grad(uh))*v*ufl.dx + 
            0.5*dt*ufl.dot(velocity_field(u_n), ufl.grad(u_n))*v*ufl.dx + 
            0.5*dt*epsilon*ufl.dot(ufl.grad(uh), ufl.grad(v))*ufl.dx +
            0.5*dt*epsilon*ufl.dot(ufl.grad(u_n), ufl.grad(v))*ufl.dx)
        
        problem = NonlinearProblem(F, uh, bcs = [bc])
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.max_it = 100  # Increase maximum number of iterations
        solver.rtol =  1e-4
        solver.report = True

        # Solve nonlinear problem
        n, converged = solver.solve(uh)
        assert (converged)
        uh.x.scatter_forward()

        smooth_vector(uh, node_patches, l)

        # Update solution at previous time step (u_n)
        u_old.x.array[:] = u_n.x.array
        u_n.x.array[:] = uh.x.array

        # Write solution to file
        xdmf_sol.write_function(u_n, t)

    xdmf_sol.close()

    print(t)
    u_exact.interpolate(lambda x: exact_solution(x, t))
    # pde.plot_pv_2d(domain, 100, epsilon, 'Epsilon Burger', 'SI_epsilon_2d_burger', location=location_figures)

    # pde.plot_pv_2d(domain, 100, u_exact, "Exact solution", "SI_exact_solution", location=location_figures)
    # pde.plot_pv_2d(domain, 100, u_n, "Approximate solution", "SI_approx_solution", location=location_figures)

    # pde.plot_pv_3d(domain, 100, u_exact, "Initial exact", "initial_exact", location=location_figures)


    print(f'Error: {np.abs(u_exact.x.array - uh.x.array)}')
