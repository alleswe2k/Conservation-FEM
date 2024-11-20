import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

import gmsh
from dolfinx.io import gmshio

from dolfinx import fem, mesh, io, plot, log
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

from dolfinx.nls.petsc import NewtonSolver

# Enable or disable real-time plotting
PLOT = True
EQ = "KPP" #Otherwise Burgers

# Creating mesh
gmsh.initialize()

membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
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
W = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, ))) # Lagrange 2 in documentation

def initial_condition(x):
    return (x[0]**2 + x[1]**2 <= 0.3) * 14*np.pi/4 + (x[0]**2 + x[1]**2 > 0.3) * np.pi/4

#def initial_condition(x, r0=0.25, x0_1=0.3, x0_2=0):
#    return 1/2*(1-np.tanh(((x[0]-x0_1)**2+(x[1]-x0_2)**2)/r0**2 - 1))

if EQ == "KPP":
    def velocity_field(u):
        return ufl.as_vector((ufl.cos(u), -ufl.sin(u)))
else: 
    def velocity_field(u):
        return ufl.as_vector((u, u))
    

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)


# Define temporal parameters
t = 0  # Start time
T = 1.0  # Final time
dt = 1e-3
num_steps = int(np.ceil(T/dt))

# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Time-dependent output

if EQ == "KPP":
    xdmf = io.XDMFFile(domain.comm, "KPP.xdmf", "w")
else:
    xdmf = io.XDMFFile(domain.comm, "burgers_equation.xdmf", "w")
xdmf.write_mesh(domain)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

# Define test function
v = ufl.TestFunction(V)

# Define the residual form F
F = (uh * v * ufl.dx 
     + 0.5 * dt * ufl.dot(velocity_field(uh), ufl.grad(uh)) * v * ufl.dx 
     - u_n * v * ufl.dx 
     + 0.5 * dt * ufl.dot(velocity_field(u_n), ufl.grad(u_n)) * v * ufl.dx)

# Define the nonlinear problem with the residual F
problem = fem.petsc.NonlinearProblem(F, uh, bcs=[bc])

# Define the nonlinear solver
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-3
solver.report = False

#Solve first time step
n, converged = solver.solve(uh)
assert (converged)


# Visualization of time dep. problem using pyvista
if PLOT:
    # pyvista.start_xvfb()

    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

    plotter = pyvista.Plotter()
    plotter.open_gif("linear_advection.gif", fps=10)

    grid.point_data["uh"] = uh.x.array
    warped = grid.warp_by_scalar("uh", factor=1)

    viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
    sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                position_x=0.1, position_y=0.8, width=0.8, height=0.1)

    renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
                                cmap=viridis, scalar_bar_args=sargs,
                                clim=[0, max(uh.x.array)])

# Updating the solution and rhs per time step
for i in range(num_steps):
    t += dt

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array

    # Solve linear problem
    n, converged = solver.solve(uh)
    assert (converged)
    uh.x.scatter_forward()
    
    #Calculate the new residual for this time step
    F = (uh * v * ufl.dx 
     + 0.5 * dt * ufl.dot(velocity_field(uh), ufl.grad(uh)) * v * ufl.dx 
     - u_n * v * ufl.dx 
     + 0.5 * dt * ufl.dot(velocity_field(u_n), ufl.grad(u_n)) * v * ufl.dx)
    
    # Redefine problem and solver with the new residual
    problem = fem.petsc.NonlinearProblem(F, uh, bcs=[bc])
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-3
    solver.report = False


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