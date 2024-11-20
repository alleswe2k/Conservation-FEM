import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

import gmsh
from dolfinx.io import gmshio

from dolfinx import fem, mesh, io, plot, nls, log
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from Utils.PDE_plot import PDE_plot

from Utils.helpers import get_nodal_h
from Utils.RV import RV

location = './Figures'
pde = PDE_plot()
PLOT = True

gmsh.initialize()

membrane = gmsh.model.occ.addRectangle(-1,-1,0,4,4)
gmsh.model.occ.synchronize()

gdim = 2
gmsh.model.addPhysicalGroup(gdim, [membrane], 1)

hmax = 1/32 # 0.05 in example
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmax)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)
gmsh.model.mesh.generate(gdim)

gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)
V = fem.functionspace(domain, ("Lagrange", 1))
DG0 = fem.functionspace(domain, ("DG", 0))

def velocity_field(u):
    # Apply nonlinear operators correctly to the scalar function u
    return ufl.as_vector([u,u])

def initial_condition(x, r0=0.25, x0_1=0.3, x0_2=0):
    return (x[0] - x0_1)**2 + (x[1] - x0_2)**2 <= r0**2

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

u_old = fem.Function(V)
u_old.name = "u_old"
u_old.interpolate(initial_condition)

CFL = 0.2
t = 0  # Start time
T = 0.5 # Final time
dt = 0.01
num_steps = int(np.ceil(T/dt))
Cvel = 0.25
CRV = 4.0

rv = RV(Cvel, CRV, domain)

# # Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Locate boundary degrees of freedom
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

RH = fem.Function(V)
RH.name = "RH"
RH.interpolate(lambda x: np.full(x.shape[1], 0, dtype = np.float64))

# Variational problem and solver
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
F = (uh*v *ufl.dx -
     u_n*v *ufl.dx + 
     0.5*dt*ufl.dot(velocity_field(uh), ufl.grad(uh))*v*ufl.dx + 
     0.5*dt*ufl.dot(velocity_field(u_n), ufl.grad(u_n))*v*ufl.dx)
pde.plot_2d(domain, 100, u_n, 'u_n', 'init_2d', location=location)

nonlin_problem = NonlinearProblem(F, uh, bcs = [bc])
nonlin_solver = NewtonSolver(MPI.COMM_WORLD, nonlin_problem)
nonlin_solver.max_it = 100  # Increase maximum number of iterations
nonlin_solver.rtol = 1e-4
nonlin_solver.report = True
if PLOT:
    # pyvista.start_xvfb()

    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

    plotter = pyvista.Plotter()
    plotter.open_gif("./Figures/burger.gif", fps=10)

    grid.point_data["uh"] = uh.x.array
    warped = grid.warp_by_scalar("uh", factor=1)

    viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
    sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                position_x=0.1, position_y=0.8, width=0.8, height=0.1)

    renderer = plotter.add_mesh(warped, show_edges=False, lighting=False,
                                cmap=viridis, scalar_bar_args=sargs,
                                clim=[0, max(uh.x.array)])
    

h_CG = get_nodal_h(domain)


#Take GFEM STEP
t += dt
n, converged = nonlin_solver.solve(uh)
assert (converged)
# uh.x.scatter_forward()

# Update solution at previous time step (u_n)
u_n.x.array[:] = uh.x.array
# Write solution to file
# xdmf.write_function(uh, t)



for i in range(num_steps -1):
    t += dt

    F_R = (RH*v*ufl.dx - 1/dt*u_n*v *ufl.dx + 1/dt*u_old*v*ufl.dx - ufl.dot(velocity_field(u_n), ufl.grad(u_n))*v*ufl.dx)
    R_problem = NonlinearProblem(F_R, RH, bcs=[bc])
    # Rh = R_problem.solve()

    Rh_problem = NewtonSolver(MPI.COMM_WORLD, R_problem)
    Rh_problem.convergence_criterion = "incremental"
    Rh_problem.max_it = 100  # Increase maximum number of iterations
    Rh_problem.rtol =  1e-4
    Rh_problem.report = True

    n, converged = Rh_problem.solve(RH)
    # n, converged = Rh.solve(uh)
    assert (converged)
    RH.x.array[:] = RH.x.array / np.max(u_n.x.array - np.mean(u_n.x.array))
    epsilon = rv.get_epsilon(uh, velocity_field, RH, h_CG)
    
    problem = NonlinearProblem(F, uh, bcs = [bc])
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.max_it = 100  # Increase maximum number of iterations
    solver.rtol =  1e-4
    solver.report = True

    # Solve nonlinear problem
    n, converged = solver.solve(uh)
    assert (converged)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_old.x.array[:] = u_n.x.array
    u_n.x.array[:] = uh.x.array

    # Write solution to file
    # xdmf.write_function(uh, t)
    # Update plot
    if PLOT:
        new_warped = grid.warp_by_scalar("uh", factor=1)
        warped.points[:, :] = new_warped.points
        warped.point_data["uh"][:] = uh.x.array
        plotter.write_frame()
if PLOT:
    plotter.close()

pde.plot_2d(domain, 100, epsilon, 'Espilon', 'epsilon_2d', location=location)
pde.plot_2d(domain, 100, RH, 'RH', 'rh_2d', location=location)