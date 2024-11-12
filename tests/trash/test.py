""" ufl artificial viscosity """
import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

import gmsh
from dolfinx.io import gmshio
from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

# Enable or disable real-time plotting
PLOT = True
# Creating mesh
gmsh.initialize()
membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
gmsh.model.occ.synchronize()

gdim = 2
gmsh.model.addPhysicalGroup(gdim, [membrane], 1)
hmax = 1 / 16
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmax)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)
gmsh.model.mesh.generate(gdim)

gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

V = fem.functionspace(domain, ("Lagrange", 1))
W = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

def initial_condition(x, r0=0.25, x0_1=0.3, x0_2=0):
    return 1 / 2 * (1 - np.tanh(((x[0] - x0_1) ** 2 + (x[1] - x0_2) ** 2) / r0 ** 2 - 1))

def velocity_field(x):
    return np.array([-2 * np.pi * x[1], 2 * np.pi * x[0]])

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

w = fem.Function(W)
w.name = "w"
w.interpolate(velocity_field)

w_values = w.x.array.reshape((-1, domain.geometry.dim))
w_inf_norm = np.linalg.norm(w_values, ord=np.inf)

# Define temporal parameters
CFL = 0.5
t = 0
T = 1.0
dt = CFL * hmax / w_inf_norm
num_steps = int(np.ceil(T / dt))

# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Time-dependent output
xdmf = io.XDMFFile(domain.comm, "linear_advection.xdmf", "w")
xdmf.write_mesh(domain)

uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

# Residual-based artificial viscosity
eps_min = 1e-10  # Minimum viscosity
eps_max = 1e-2   # Maximum viscosity

# Define the variational problem with artificial viscosity
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(0))

# Formulate the residual-based viscosity
def artificial_viscosity(u_n, w, h):
    grad_u_n = ufl.grad(u_n)
    residual = ufl.div(w * u_n) + ufl.dot(w, grad_u_n)
    viscosity = h * ufl.sqrt(ufl.inner(residual, residual))  # Residual-based viscosity scaling
    return ufl.max_value(eps_min, ufl.min_value(viscosity, eps_max))  # Clamp viscosity values

# Set up bilinear and linear forms
a = u * v * ufl.dx + 0.5 * dt * ufl.dot(w, ufl.grad(u)) * v * ufl.dx
L = u_n * v * ufl.dx - 0.5 * dt * ufl.dot(w, ufl.grad(u_n)) * v * ufl.dx

bilinear_form = fem.form(a)
linear_form = fem.form(L)

# Preassemble A
A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
b = create_vector(linear_form)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

if PLOT:
    # pyvista.start_xvfb()

    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

    plotter = pyvista.Plotter()
    plotter.open_gif("RV.gif", fps=10)

    grid.point_data["uh"] = uh.x.array
    warped = grid.warp_by_scalar("uh", factor=1)

    viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
    sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                position_x=0.1, position_y=0.8, width=0.8, height=0.1)

    renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
                                cmap=viridis, scalar_bar_args=sargs,
                                clim=[0, max(uh.x.array)])

# Main time-stepping loop
for i in range(num_steps):
    t += dt

    # Update artificial viscosity at each timestep
    h = fem.Constant(domain, PETSc.ScalarType(hmax))
    epsilon = artificial_viscosity(u_n, w, h)

    # Update forms with artificial viscosity
    a = u * v * ufl.dx + 0.5 * dt * ufl.dot(w, ufl.grad(u)) * v * ufl.dx + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx - 0.5 * dt * ufl.dot(w, ufl.grad(u_n)) * v * ufl.dx + epsilon * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * ufl.dx

    bilinear_form = fem.form(a)
    linear_form = fem.form(L)

    # Reassemble only b, as A remains the same
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # Solve the linear system
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
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
