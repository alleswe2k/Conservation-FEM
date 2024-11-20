import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

import gmsh
from dolfinx.io import gmshio
from dolfinx import plot

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, LinearProblem
from mpi4py import MPI

import matplotlib.pyplot as plt
import numpy as np
import pyvista

import basix.ufl
import dolfinx
import ufl


def initial_condition(x, r0=0.25, x0_1=0.3, x0_2=0):
    return (x[0] - x0_1)**2 + (x[1] - x0_2)**2 <= r0**2


shift = 0.25
# nodes = np.array([[0.0, 0.0], [shift, 0.0], [1.0, 0.0], 
#                   [0.0, 0.5], [shift, 0.5], [1.0, 0.5],
#                   [0.0, 1.0], [shift, 1.0], [1.0, 1.0]], dtype=np.float64)
# connectivity = np.array([[0, 1, 3], [1, 3, 4], [1, 2, 4], [2, 4, 5],
#                          [3, 4, 6], [4, 6, 7], [4, 5, 7], [5, 7, 8]], dtype=np.int64)
nodes = np.array([[0.0, 0.0], [shift, 0.0], [1.0, 0.0], 
                  [0.0, 0.5], [shift, 0.5], [1.0, 0.5]], dtype=np.float64)
connectivity = np.array([[0, 1, 3], [1, 3, 4], [1, 2, 4], [2, 4, 5]], dtype=np.int64)
c_el = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(nodes.shape[1],)))
domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, connectivity, nodes, c_el)
# domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [2, 2], cell_type=mesh.CellType.triangle)
hmax = 1/16 # 0.05 in example


#domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 4, mesh.CellType.quadrilateral)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

domain.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

V = fem.functionspace(domain, ("Lagrange", 1))
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

# Define temporal parameters
CFL = 0.5
t = 0  # Start time
T = 1.0  # Final time
dt = CFL*hmax/w_inf_norm
num_steps = int(np.ceil(T/dt))
Cvel = 0.25
CRV = 1.0

# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Time-dependent output
xdmf = io.XDMFFile(domain.comm, "RV_node.xdmf", "w")
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
# NOTE: Approach 0 seems to be correct
approach = 0
if approach == 0:
    h_DG = fem.Function(DG0)  # Cell-based function for hk values

    # Access mesh topology and geometry
    cell_to_vertex_map = domain.topology.connectivity(domain.topology.dim, 0)
    vertex_coords = domain.geometry.x

    # Compute hk values (e.g., min of edges)
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    hk_values = np.zeros(num_cells)

    for cell in range(num_cells):
        # Get the vertices of the current cell
        cell_vertices = cell_to_vertex_map.links(cell)
        coords = vertex_coords[cell_vertices]  # Coordinates of the vertices
        
        # Compute the lengths of the edges
        edges = [np.linalg.norm(coords[i] - coords[j]) for i in range(len(coords)) for j in range(i + 1, len(coords))]
        hk_values[cell] = min(edges)  # Compute min edge length for the cell
        print("Edges", edges)
        print("Coords", coords)
        print("hmin", hk_values[cell])

    # Assign hk values to h_DG
    h_DG.x.array[:] = hk_values
    for h in h_DG.x.array:
        print(h)

    # Project cell-based h_DG to nodal-based h_CG
    v = ufl.TestFunction(V)
    h_trial = ufl.TrialFunction(V)
    a_h = h_trial * v * ufl.dx
    L_h = h_DG * v * ufl.dx

    # Solve linear system to project h_DG onto V
    problem = LinearProblem(a_h, L_h, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    h_CG = problem.solve()  # Resulting function in V (nodal-based)
    print(h_CG.x.array)
elif approach == 1:
    h_DG = fem.Function(DG0)  # Cell-based function for hk values

    # Compute hk values (e.g., min of edges)
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    hk_values = np.zeros(num_cells)

    for cell in range(num_cells):
        loc2glb = DG0.dofmap.cell_dofs(cell)  # DOFs for DG0 are cell-based
        x = DG0.tabulate_dof_coordinates()[loc2glb]  # Coordinates of cell DOFs
        x = x[0]
        edges = [np.linalg.norm(x[i] - x[j]) for i in range(3) for j in range(i + 1, 3)]
        hk_values[cell] = min(edges)  # Compute min edge length for cell

    # Assign hk values to h_DG
    h_DG.x.array[:] = hk_values

    # Project cell-based h_DG to nodal-based h_CG
    v = ufl.TestFunction(V)
    h_trial = ufl.TrialFunction(V)
    a_h = h_trial * v * ufl.dx
    L_h = h_DG * v * ufl.dx

    # Solve linear system to project h_DG onto V
    problem = LinearProblem(a_h, L_h, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    h_CG = problem.solve()  # Resulting function in V (nodal-based)
elif approach == 2:
    h_DG = fem.Function(DG1)
    num_cells = domain.topology.index_map(domain.topology.dim).size_local

    for cell in range(num_cells):
        # TODO: DG instead of V?
        loc2glb = DG1.dofmap.cell_dofs(cell)
        x = DG1.tabulate_dof_coordinates()[loc2glb]
        edges = [np.linalg.norm(x[i] - x[j]) for i in range(3) for j in range(i+1, 3)]
        hk = min(edges) # NOTE: Max gives better convergence
        # print(f'{np.round(x)} ')
        # print(hk)
        h_DG.x.array[loc2glb] = hk

    for h in h_DG.x.array:
        print(h)
    v = ufl.TestFunction(V)

    h_trial = ufl.TrialFunction(V)
    a_h = h_trial * v * ufl.dx
    L_h = h_DG * v * ufl.dx

    # Solve linear system
    problem = LinearProblem(a_h, L_h, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    h_CG = problem.solve() # returns dolfinx.fem.Function

    print(h_CG.x.array)

""" Alternative: Sqrt of element k """
# h_DG = fem.Function(DG0)
# v = ufl.TestFunction(DG0)
# v_integral = v * ufl.dx
# b = create_vector(fem.form(v_integral))
# assemble_vector(b, fem.form(v_integral))
# b_array = b.array  # PETSc vector values as a NumPy array
# print(b_array)
# hk_values = np.sqrt(2 * b_array / np.sqrt(3))

# for cell_id, hk in enumerate(hk_values):
#     print(f"Cell {cell_id}: h_k = {hk}")

# h_DG.x.array[:] = hk_values

# # Project
# v = ufl.TestFunction(V)

# h_trial = ufl.TrialFunction(V)
# a_h = h_trial * v * ufl.dx
# L_h = h_DG * v * ufl.dx

# # Solve linear system
# problem = LinearProblem(a_h, L_h, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
# h_CG = problem.solve() # returns dolfinx.fem.Function

dof_coordinates = V.tabulate_dof_coordinates()
unique_dofs, indices = np.unique(dof_coordinates, axis=0, return_index=True)
h_values = h_CG.x.array[indices]

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True, color="white")

plotter.add_point_labels(
    unique_dofs,
    [f"{val:.2f}" for val in h_values],  # Format values for readability
    point_size=20,
    font_size=24,
    text_color="blue",
)

plotter.view_xy()

if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    plotter.screenshot("h_CG_nodal_values.png")