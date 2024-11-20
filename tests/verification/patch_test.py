import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

import gmsh
from dolfinx.io import gmshio

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, LinearProblem

domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [2, 2], cell_type=mesh.CellType.triangle, diagonal=mesh.DiagonalType.crossed)

hmax = 1/16
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
CRV = 1.0
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

""" First, project hk in DG(0) on h_h in Lagrange(1) """
h_DG = fem.Function(DG1)
num_cells = domain.topology.index_map(domain.topology.dim).size_local

for cell in range(num_cells):
    # TODO: DG instead of V?
    loc2glb = DG1.dofmap.cell_dofs(cell)
    x = DG1.tabulate_dof_coordinates()[loc2glb]
    edges = [np.linalg.norm(x[i] - x[j]) for i in range(3) for j in range(i+1, 3)]
    hk = min(edges) # NOTE: Max gives better convergence
    h_DG.x.array[loc2glb] = hk

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

# Iterate over each node and compute epsilon based on the patch
for node, adjacent_nodes in node_patches.items():
    print("Node:", node, " - Adjacent nodes:", [int(adjacent_node) for adjacent_node in adjacent_nodes])

""" Assemble stiffness matrix, obtain element values """
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
A = assemble_matrix(fem.form(a), bcs=[bc])
A.assemble()

# Grid visualization
# tdim = tdim = domain.topology.dim
# domain.topology.create_connectivity(tdim, tdim)
# topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
# grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, show_edges=True)
# plotter.view_xy()
# if not pyvista.OFF_SCREEN:
#     plotter.show()
# else:
#     figure = plotter.screenshot("fundamentals_mesh.png")
# Extract DoF coordinates
dof_coordinates = V.tabulate_dof_coordinates()

# Get unique node indices (for non-overlapping partitions)
node_indices = np.arange(len(dof_coordinates))

# Prepare mesh for PyVista
topology, cell_types, geometry = plot.vtk_mesh(domain, domain.topology.dim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Add labels for node indices
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True, color="white")
plotter.add_point_labels(dof_coordinates, node_indices.astype(str), point_size=20, font_size=24, text_color="black")
plotter.view_xy()

# Show or save the plot
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    plotter.screenshot("labeled_mesh.png")