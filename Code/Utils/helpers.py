import ufl
import numpy as np

from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

def get_nodal_h(domain):
    DG0 = fem.functionspace(domain, ("DG", 0))
    V = fem.functionspace(domain, ("Lagrange", 1))
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
    lin_problem = LinearProblem(a_h, L_h, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    h_CG = lin_problem.solve() # returns dolfinx.fem.Function

    return h_CG

def smooth_vector(domain, u, patches):
    l = 4    

    for node, adjacent_nodes in patches.items():
        # node = i and adjacent_nodes (including self) = j
        # print("Node:", node, " - Adjacent nodes:", adjacent_nodes)
        sum = 0
        for adj_node in adjacent_nodes:
            if node != adj_node:
                sum += u.x.array[adj_node]

        d = len(adjacent_nodes) - 1
        u.x.array[node] = (sum + (l-1)*d*u.x.array[node]) / (l*d)