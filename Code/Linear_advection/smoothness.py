""" TODO: residual and artificial viscosity plot """
import matplotlib as mpl
import pyvista
import ufl
import numpy as np
import os

from petsc4py import PETSc
from mpi4py import MPI

import gmsh
from dolfinx.io import gmshio

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, LinearProblem

from PDE_solver import PDE_solver

pde_solve = PDE_solver()
# print(PETSc.ScalarType)

# Enable or disable real-time plotting
PLOT = False
# Creating mesh
gmsh.initialize()

membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
gmsh.model.occ.synchronize()

gdim = 2
gmsh.model.addPhysicalGroup(gdim, [membrane], 1)

fraction = 8
hmax = 1/fraction # 0.05 in example
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmax)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)
gmsh.model.mesh.generate(gdim)

gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

V = fem.functionspace(domain, ("Lagrange", 1))
# domain.geometry.dim = (2, )
W = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, ))) # Lagrange 2 in documentation
DG0 = fem.functionspace(domain, ("DG", 0))
DG1 = fem.functionspace(domain, ("DG", 1))

# def initial_condition(x, r0=0.25, x0_1=0.3, x0_2=0):
#     return 1/2*(1-np.tanh(((x[0]-x0_1)**2+(x[1]-x0_2)**2)/r0**2 - 1))
def initial_condition(x, r0=0.25, x0_1=0.3, x0_2=0):
    return (x[0] - x0_1)**2 + (x[1] - x0_2)**2 <= r0**2

def velocity_field(x):
    return np.array([-2*np.pi*x[1], 2*np.pi*x[0]])

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

u_old = fem.Function(V)
u_old.name = "u_old"
u_old.interpolate(initial_condition)

u_ex = fem.Function(V)
u_ex.interpolate(initial_condition)

# velocity field f_prim
w = fem.Function(W)
w.name = "w"
w.interpolate(velocity_field)

w_values = w.x.array.reshape((-1, domain.geometry.dim))
w_inf_norm = np.linalg.norm(w_values, ord=np.inf)
# TODO: This is probably more correct
# w_norms = np.linalg.norm(w_values, axis=1)
# w_inf_norm = np.max(w_norms)

# Define temporal parameters
CFL = 0.5
t = 0  # Start time
T = 1.0  # Final time
dt = CFL*hmax/w_inf_norm
num_steps = int(np.ceil(T/dt))
Cvel = 0.25
Cm = 0.5

# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Time-dependent output
xdmf = io.XDMFFile(domain.comm, "smoothness.xdmf", "w")
xdmf.write_mesh(domain)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
# xdmf.write_function(uh, t)

# Variational problem and solver
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(0))

a = u * v * ufl.dx + 0.5 * dt * ufl.dot(w, ufl.grad(u)) * v * ufl.dx
L = u_n * v * ufl.dx - 0.5 * dt * ufl.dot(w, ufl.grad(u_n)) * v * ufl.dx

# Preparing linear algebra structures for time dep. problems
bilinear_form = fem.form(a)
linear_form = fem.form(L)

# A does not change through time, but b does
A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
b = create_vector(linear_form)

# Can no longer use LinearProblem to solve since we already
# assembled a into matrix A. Therefore, create linear algebra solver with petsc4py
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# Visualization of time dep. problem using pyvista
if PLOT:
    # pyvista.start_xvfb()

    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

    plotter = pyvista.Plotter()
    plotter.open_gif("smoothness.gif", fps=10)

    grid.point_data["uh"] = uh.x.array
    warped = grid.warp_by_scalar("uh", factor=1)

    viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
    sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                position_x=0.1, position_y=0.8, width=0.8, height=0.1)

    renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
                                cmap=viridis, scalar_bar_args=sargs,
                                clim=[0, max(uh.x.array)])

""" First, project hk in DG(0) on h_h in Lagrange(1) """
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
problem = LinearProblem(a_h, L_h, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
h_CG = problem.solve() # returns dolfinx.fem.Function

""" Creat patch dictionary """
# Dictionary to store adjacent nodes for each node
node_patches = {}

# Loop over each cell to build node adjacency information
for cell in range(domain.topology.index_map(domain.topology.dim).size_local):
    cell_nodes = V.dofmap.cell_dofs(cell)
    for node in cell_nodes:
        if node not in node_patches:
            node_patches[node] = set()
        # Add all other nodes in this cell to the patch of the current node
        # node_patches[node].update(n for n in cell_nodes if n != node)
        node_patches[node].update(n for n in cell_nodes)

""" Assemble stiffness matrix, obtain element values """
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
A = assemble_matrix(fem.form(a), bcs=[bc])
A.assemble()

""" Take on GFEM step for residual calculation """
t += dt

# Update the right hand side reusing the initial vector
with b.localForm() as loc_b:
    loc_b.set(0)
assemble_vector(b, linear_form)

# Apply Dirichlet boundary condition to the vector
apply_lifting(b, [bilinear_form], [[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc])

# Solve linear problem
solver.solve(b, uh.x.petsc_vec)
uh.x.scatter_forward()

# Update solution at previous time step (u_n)
u_n.x.array[:] = uh.x.array

# Write solution to file
xdmf.write_function(uh, t)

# """ Then time loop """
for i in range(num_steps-1):
    t += dt

    a_R = u * v * ufl.dx
    L_R = 1/dt * u_n * v * ufl.dx - 1/dt * u_old * v * ufl.dx + ufl.dot(w, ufl.grad(u_n)) * v * ufl.dx

    # Solve linear system
    problem = LinearProblem(a_R, L_R, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    Rh = problem.solve() # returns dolfinx.fem.Function
    Rh.x.array[:] = Rh.x.array / np.max(u_n.x.array - np.mean(u_n.x.array))

    epsilon = fem.Function(V)

    for node, adjacent_nodes in node_patches.items():
        # node = i and adjacent_nodes (including self) = j
        # print("Node:", node, " - Adjacent nodes:", adjacent_nodes)
        hi = h_CG.x.array[node]
        w_values = w.x.array.reshape((-1, domain.geometry.dim))
        fi = w_values[node]
        fi_norm = np.linalg.norm(fi)

        numerator = 0
        denominator = 0
        for adj_node in adjacent_nodes:
            # print(adj_node)
            # print(A.getValue(node, adj_node))
            beta = A.getValue(node, adj_node)
            numerator += beta * (u_n.x.array[adj_node] - u_n.x.array[node])
            denominator += np.abs(beta) * np.abs(u_n.x.array[adj_node] - u_n.x.array[node])

        alpha = np.abs(numerator) / max(denominator, 1e-8)
        # print('Numerator:', np.abs(numerator), ' - Denominator:', denominator, ' - Alpha:', alpha)
        epsilon.x.array[node] = alpha * Cm * hi * fi_norm

    a = u * v * ufl.dx + 0.5 * dt * ufl.dot(w, ufl.grad(u)) * v * ufl.dx + 0.5 * epsilon * dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx - 0.5 * dt * ufl.dot(w, ufl.grad(u_n)) * v * ufl.dx - 0.5 * epsilon * dt * ufl.dot(ufl.grad(u_n), ufl.grad(v)) * ufl.dx

    # Preparing linear algebra structures for time dep. problems
    bilinear_form = fem.form(a)
    linear_form = fem.form(L)

    # A does not change through time, but b does
    A = assemble_matrix(bilinear_form, bcs=[bc])
    A.assemble()
    b = create_vector(linear_form)

    solver.setOperators(A)
    """ Rest """
    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])
    
    # Solve linear problem
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Save previous solution
    u_old.x.array[:] = u_n.x.array
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

error_L2 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
if domain.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")

location = "./Figures/linear_advection/SI"
pde_solve.plot_2d(domain, fraction, epsilon, 'Espilon', 'epsilon_2d', location=location)
pde_solve.plot_solution(domain, fraction, Rh, 'Rh', 'rv', location=location)