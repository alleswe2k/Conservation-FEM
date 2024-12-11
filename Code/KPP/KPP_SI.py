import matplotlib as mpl
import pyvista
import ufl
import numpy as np
from tqdm import tqdm

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_matrix, NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import gmshio
import gmsh

from Utils.helpers import get_nodal_h
from Utils.SI import SI
from Utils.PDE_plot import PDE_plot

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
location_figures = os.path.join(script_dir, 'Figures/SI') # location = './Figures'
location_data = os.path.join(script_dir, 'Data/SI') # location = './Data'

pde = PDE_plot()
PLOT = False

gmsh.initialize()

membrane = gmsh.model.occ.addRectangle(-2,-2,0,4,4)
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

def initial_condition(x):
    return (x[0]**2 + x[1]**2 <= 1) * 14*np.pi/4 + (x[0]**2 + x[1]**2 > 1) * np.pi/4

def velocity_field(u):
    # Apply nonlinear operators correctly to the scalar function u
    return ufl.as_vector([ufl.cos(u), -ufl.sin(u)])


u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

u_old = fem.Function(V)
u_old.name = "u_old"
u_old.interpolate(initial_condition)

plot_func = fem.Function(V)
plot_func.name = "plot_func"

CFL = 0.2
t = 0  # Start time
T = 1.0 # Final time
dt = 0.01
num_steps = int(np.ceil(T/dt))
Cm = 0.5

si = SI(Cm, domain, 1e-8)

""" Creat patch dictionary """
node_patches = si.get_patch_dictionary()


# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(np.pi/4), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Time-dependent output
xdmf_alpha = io.XDMFFile(domain.comm, f"{location_data}/alpha_sigmoid.xdmf", "w")
xdmf_alpha.write_mesh(domain)
xdmf_epsilon = io.XDMFFile(domain.comm, f"{location_data}/epsilon_sigmoid.xdmf", "w")
xdmf_epsilon.write_mesh(domain)
xdmf_sol = io.XDMFFile(domain.comm, f"{location_data}/sol.xdmf", "w")
xdmf_sol.write_mesh(domain)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

# Variational problem and solver
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

if PLOT:
    # pyvista.start_xvfb()

    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

    plotter = pyvista.Plotter()
    plotter.open_gif(f"{location_figures}/SI_E_burger.gif", fps=10)

    grid.point_data["uh"] = uh.x.array
    warped = grid.warp_by_scalar("uh", factor=1)

    viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
    sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                position_x=0.1, position_y=0.8, width=0.8, height=0.1)

    renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
                                cmap=viridis, scalar_bar_args=sargs,
                                clim=[0, max(uh.x.array)])
    

h_CG = get_nodal_h(domain)

counter = 0
for i in tqdm(range(num_steps)):
    t += dt

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

    # Update solution at previous time step (u_n)
    u_old.x.array[:] = u_n.x.array
    u_n.x.array[:] = uh.x.array

    # Write solution to file
    xdmf_alpha.write_function(plot_func, t)
    xdmf_epsilon.write_function(epsilon, t)
    xdmf_sol.write_function(u_n, t)
    #     counter = 0
    # counter += 1

    # Update plot
    if PLOT:
        new_warped = grid.warp_by_scalar("uh", factor=1)
        warped.points[:, :] = new_warped.points
        warped.point_data["uh"][:] = uh.x.array
        plotter.write_frame()


xdmf_alpha.close()
xdmf_epsilon.close()
xdmf_sol.close()

#pde.plot_pv_2d(domain, 100, epsilon, 'Epsilon Burger', 'SI_epsilon_2d_burger', location=location_figures)

# pde.plot_pv_2d(domain, 100, u_exact, "Exact solution", "SI_exact_solution", location=location_figures)
# pde.plot_pv_2d(domain, 100, u_n, "Approximate solution", "SI_approx_solution", location=location_figures)

# pde.plot_pv_3d(domain, 100, u_exact, "Initial exact", "initial_exact", location=location_figures)


if PLOT:
    plotter.close()
