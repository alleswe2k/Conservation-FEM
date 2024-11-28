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
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, LinearProblem

from Utils.PDE_plot import PDE_plot
from Utils.PDE_realtime_plot import PDE_realtime_plot
from Utils.SI import SI
from Utils.helpers import get_nodal_h

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
location_figures = os.path.join(script_dir, 'Figures/SI')
location_data = os.path.join(script_dir, 'Data/SI/solution.xdmf')

import dolfinx
import basix.ufl

# shift = 0.25
# nodes = np.array([[0.0, 0.0], [shift, 0.0], [1.0, 0.0], 
#                   [0.0, 0.5], [shift, 0.5], [1.0, 0.5]], dtype=np.float64)
# connectivity = np.array([[0, 1, 3], [1, 3, 4], [1, 2, 4], [2, 4, 5]], dtype=np.int64)
# c_el = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(nodes.shape[1],)))
# domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, connectivity, nodes, c_el)
# domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [2, 2], cell_type=mesh.CellType.triangle)
hmax = 1/16 # 0.05 in example
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-1, -1]), np.array([1, 1])], [1, 1], cell_type=mesh.CellType.triangle, diagonal=mesh.DiagonalType.crossed)


#domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 4, mesh.CellType.quadrilateral)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

domain.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

V = fem.functionspace(domain, ("Lagrange", 1))
# domain.geometry.dim = (2, )
W = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, ))) # Lagrange 2 in documentation
DG0 = fem.functionspace(domain, ("DG", 0))

bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

""" KPP IC """
# def initial_condition(x, r0=0.25, x0_1=0.3, x0_2=0):
#     return ((x[0] - x0_1)**2 + (x[1] - x0_2)**2 <= r0**2) * 14*np.pi + ((x[0] - x0_1)**2 + (x[1] - x0_2)**2 > r0**2) * np.pi / 4
""" Discont. IC """
# def initial_condition(x, r0=0.25, x0_1=0.3, x0_2=0):
#     return (x[0] - x0_1)**2 + (x[1] - x0_2)**2 <= r0**2
""" Cont. IC """
def initial_condition(x, r0=0.25, x0_1=0.3, x0_2=0):
    return 1/2*(1-np.tanh(((x[0]-x0_1)**2+(x[1]-x0_2)**2)/r0**2 - 1))

def velocity_field(x):
    return np.array([-2*np.pi*x[1], 2*np.pi*x[0]])

si = SI(0.5, domain, 1e-8)

""" Creat patch dictionary """
# Dictionary to store adjacent nodes for each node
node_patches = si.get_patch_dictionary()

""" Assemble stiffness matrix, obtain element values """
# TODO: Verify this somehow with small mesh
a_stiffness = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
stiffness_matrix = assemble_matrix(fem.form(a_stiffness))
stiffness_matrix.assemble()
dense_matrix = stiffness_matrix.copy().getValuesCSR()
rows, cols, values = dense_matrix
# Convert to sparse format for better printing
from scipy.sparse import csr_matrix
scipy_sparse_matrix = csr_matrix((values, cols, rows))

# Print sparse matrix
print("Sparse matrix (CSR format):")
print(scipy_sparse_matrix)

# If dense output is desired (use only for small matrices)
print("\nDense matrix:")
dense_matrix = scipy_sparse_matrix.toarray()
print(dense_matrix)

for node, adjacent_nodes in node_patches.items():
        print("----", node, "----")
        for adj_node in adjacent_nodes:
            beta = stiffness_matrix.getValue(node, adj_node)
            print(adj_node, beta)

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