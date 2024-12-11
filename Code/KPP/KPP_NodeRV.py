import matplotlib as mpl
import pyvista
import ufl
import numpy as np
import os 
from tqdm import tqdm

from petsc4py import PETSc
from mpi4py import MPI

import gmsh
from dolfinx.io import gmshio

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver

from Utils.PDE_plot import PDE_plot
from Utils.RV import RV
from Utils.SI import SI
from Utils.helpers import get_nodal_h

script_dir = os.path.dirname(os.path.abspath(__file__))

location_fig = os.path.join(script_dir, 'Figures/RV') # location = './Figures'
location_data = os.path.join(script_dir, 'Data') # location = './Figures'

pde = PDE_plot()
PLOT = True
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

u_old_old = fem.Function(V)
u_old_old.name = "u_old_old"
u_old_old.interpolate(initial_condition)


CFL = 0.5
t = 0  # Start time
T = 1.0  # Final time
dt = 0.01
num_steps = int(np.ceil(T/dt))
Cvel = 0.5
CRV = 4.0

rv = RV(Cvel, CRV, domain)
si = SI(1, domain, 1e-8 )
node_patches = si.get_patch_dictionary()

# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(np.pi/4), fem.locate_dofs_topological(V, fdim, boundary_facets), V)
bc0 = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Time-dependent output
xdmf = io.XDMFFile(domain.comm, location_data + '/KPP_RV.xdmf', "w")
xdmf.write_mesh(domain)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

RH = fem.Function(V)
RH.name = "RH"
RH.interpolate(lambda x: np.full(x.shape[1], 0, dtype = np.float64))

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

if PLOT:
    # pyvista.start_xvfb()

    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

    plotter = pyvista.Plotter()
    plotter.open_gif(location_fig+"/KPP.gif", fps=10)

    grid.point_data["uh"] = uh.x.array
    warped = grid.warp_by_scalar("uh", factor=1)

    viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
    sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                position_x=0.1, position_y=0.8, width=0.8, height=0.1)

    renderer = plotter.add_mesh(warped, show_edges=False, lighting=False,
                                cmap=viridis, scalar_bar_args=sargs,
                                clim=[0, max(uh.x.array)])

    

h_CG = get_nodal_h(domain)

for i in tqdm(range(num_steps)):
    t += dt

    """ BDF2 """
    F_R = (RH*v*ufl.dx - 
           3/(2*dt)*u_n*v *ufl.dx +
           4/(2*dt)*u_old*v*ufl.dx - 
           1/(2*dt)*u_old_old*v*ufl.dx - 
           ufl.dot(velocity_field(u_n), ufl.grad(u_n))*v*ufl.dx)
    R_problem = NonlinearProblem(F_R, RH, bcs=[bc0])
    # Rh = R_problem.solve()

    Rh_problem = NewtonSolver(MPI.COMM_WORLD, R_problem)
    Rh_problem.convergence_criterion = "incremental"
    Rh_problem.max_it = 100  # Increase maximum number of iterations
    Rh_problem.rtol =  1e-1
    Rh_problem.report = True

    n, converged = Rh_problem.solve(RH)

    epsilon = rv.get_epsilon_nonlinear(uh, u_n, velocity_field, RH, h_CG, node_patches)
 
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

    # Solve linear problem
    n, converged = solver.solve(uh)
    assert (converged)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_old_old.x.array[:] = u_old.x.array
    u_old.x.array[:] = u_n.x.array
    u_n.x.array[:] = uh.x.array

    # Write solution to file
    xdmf.write_function(uh, t)
    # Update plot
    if PLOT:
        new_warped = grid.warp_by_scalar("uh", factor=1)
        warped.points[:, :] = new_warped.points
        warped.point_data["uh"][:] = uh.x.array
        plotter.write_frame()
if PLOT:
    plotter.close()
xdmf.close()

pde.plot_pv_3d(domain, int(1/hmax), u_n, 'Solution uh', 'uh_3d', location=location_fig)
pde.plot_pv_2d(domain, int(1/hmax), epsilon, 'Espilon', 'epsilon_2d', location=location_fig)
RH.x.array[:] = np.abs(RH.x.array)
pde.plot_pv_2d(domain, int(1/hmax), RH, 'RH', 'Rh_2d', location=location_fig)