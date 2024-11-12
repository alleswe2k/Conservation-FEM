""" Artificial viscosity function """
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
# def initial_condition(x, r0=0.25, x0_1=0.3, x0_2=0):
#     return (x[0] - x0_1)**2 + (x[1] - x0_2)**2 <= r0**2

def velocity_field(x):
    return np.array([-2 * np.pi * x[1], 2 * np.pi * x[0]])

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

u_old = fem.Function(V)
u_old.name = "u_old"
u_old.interpolate(initial_condition)

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

# Parameters for artificial viscosity
Cvel = 0.25
CRV = 1.0

# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Time-dependent output
xdmf = io.XDMFFile(domain.comm, "linear_advection_rv.xdmf", "w")
xdmf.write_mesh(domain)

uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

# Variational problem
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(0))

# Artificial viscosity function based on element-wise residuals
def compute_artificial_viscosity(u_n, u_old, w, h):
    # Compute residual R(Un) for each element
    residual = (u_n - u_old) / dt + ufl.dot(w, ufl.grad(u_n))

    # Compute wave speed beta_K per element
    beta_K = ufl.sqrt(ufl.dot(ufl.grad(w[0]), ufl.grad(w[0])) + ufl.dot(ufl.grad(w[1]), ufl.grad(w[1])))

    # Average solution value for normalization
    volume = fem.assemble_scalar(fem.form(fem.Constant(domain, PETSc.ScalarType(1)) * ufl.dx))  # Total volume of the domain
    u_avg = fem.assemble_scalar(fem.form(u_n * ufl.dx)) / volume

    # Residual norm
    residual_norm = ufl.sqrt(ufl.inner(residual, residual))

    # Define viscosity with min-max clamping to avoid negative or excessive values
    viscosity_expr = ufl.min_value(
        Cvel * h * beta_K,
        CRV * h ** 2 * residual_norm / ufl.sqrt((u_n - u_avg)**2)
    )

    # Project viscosity_expr to a function space to apply element-wise
    V_viz = fem.functionspace(domain, ("DG", 0))  # DG-0 space for cell-wise constant functions
    epsilon_func = fem.Function(V_viz)
    epsilon_func.interpolate(fem.Expression(viscosity_expr, V_viz.element.interpolation_points()))
    return epsilon_func

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

""" One GFEM step! """
t += dt

# Set up bilinear and linear forms
a = u * v * ufl.dx + 0.5 * dt * ufl.dot(w, ufl.grad(u)) * v * ufl.dx
L = u_n * v * ufl.dx - 0.5 * dt * ufl.dot(w, ufl.grad(u_n)) * v * ufl.dx

# Formulate the variational problem and linear algebra structures
bilinear_form = fem.form(a)
linear_form = fem.form(L)

# Assemble matrix A (does not change over time)
A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
b = create_vector(linear_form)

# Solve linear system
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

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
# Main time-stepping loop
for i in range(num_steps - 1):
    t += dt

    # Update artificial viscosity at each timestep
    h = fem.Constant(domain, PETSc.ScalarType(hmax))
    epsilon_func = compute_artificial_viscosity(u_n, u_old, w, h)
    u_old.x.array[:] = u_n.x.array

    # Set up bilinear and linear forms with cell-wise artificial viscosity
    a = (u * v * ufl.dx + 0.5 * dt * ufl.dot(w, ufl.grad(u)) * v * ufl.dx +
         0.5 * epsilon_func * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = (u_n * v * ufl.dx - 0.5 * dt * ufl.dot(w, ufl.grad(u_n)) * v * ufl.dx -
         0.5 * epsilon_func * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * ufl.dx)

    # Formulate the variational problem and linear algebra structures
    bilinear_form = fem.form(a)
    linear_form = fem.form(L)

    # Assemble matrix A (does not change over time)
    A = assemble_matrix(bilinear_form, bcs=[bc])
    A.assemble()
    b = create_vector(linear_form)

    # Solve linear system
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

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
