import ufl
import numpy as np

from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
class SI:
    def __init__(self, Cm, domain):
        self.Cm = Cm
        self.domain = domain
    
    def get_patch_dictionary(self):
        """ Creat patch dictionary """
        V = fem.functionspace(self.domain, ("Lagrange", 1))
        # Dictionary to store adjacent nodes for each node
        node_patches = {}

        # Loop over each cell to build node adjacency information
        for cell in range(self.domain.topology.index_map(self.domain.topology.dim).size_local):
            cell_nodes = V.dofmap.cell_dofs(cell)
            for node in cell_nodes:
                if node not in node_patches:
                    node_patches[node] = set()
                # Add all other nodes in this cell to the patch of the current node
                # node_patches[node].update(n for n in cell_nodes if n != node)
                node_patches[node].update(n for n in cell_nodes)
        
        return node_patches

    def get_epsilon_linear(self, w, node_patches, h_CG, u_n, stiffness_matrix):
        V = fem.functionspace(self.domain, ("Lagrange", 1))
        epsilon = fem.Function(V)
        # alphas = []

        for node, adjacent_nodes in node_patches.items():
            # node = i and adjacent_nodes (including self) = j
            # print("Node:", node, " - Adjacent nodes:", adjacent_nodes)
            hi = h_CG.x.array[node]
            w_values = w.x.array.reshape((-1, self.domain.geometry.dim))
            fi = w_values[node]
            fi_norm = np.linalg.norm(fi)

            numerator = 0
            denominator = 0
            G_max = 0 # max |G_j|
            for adj_node in adjacent_nodes:
                G_max = max(G_max, np.abs(u_n.x.array[adj_node]))
                delta_u = u_n.x.array[adj_node] - u_n.x.array[node]
                # print(adj_node)
                # print(A.getValue(node, adj_node))
                beta = stiffness_matrix.getValue(node, adj_node)
                # beta = 1
                numerator += beta * (delta_u)
                denominator += np.abs(beta) * np.abs(delta_u)

            eps = 1e-8 * G_max
            # print(max(denominator, eps, 1e-4))
            # if max(denominator, eps) == 0:
            #     alpha = 0
            # else:
            #     alpha = np.abs(numerator) / max(denominator, eps)
            alpha = np.abs(numerator) / max(denominator, eps, 1e-2)
            # print(alpha, self.Cm, hi, fi_norm)
            epsilon.x.array[node] = alpha * self.Cm * hi * fi_norm
            # alphas.append(alpha)
        
        # print(max(alphas), min(alphas))
        
        return epsilon

    def get_epsilon(self, velocity_field, node_patches, h_CG, u_n, A):
        V = fem.functionspace(self.domain, ("Lagrange", 1))
        epsilon = fem.Function(V)

        for node, adjacent_nodes in node_patches.items():
            # node = i and adjacent_nodes (including self) = j
            # print("Node:", node, " - Adjacent nodes:", adjacent_nodes)
            hi = h_CG.x.array[node]
            w = velocity_field(u_n.x.array[node])
            fi = np.array(w, dtype = 'float')
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
            epsilon.x.array[node] = alpha * self.Cm * hi * fi_norm
        
        return epsilon