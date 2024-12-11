import ufl
import numpy as np

from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
class SI:
    def __init__(self, Cm, domain, eps):
        self.Cm = Cm
        self.domain = domain
        self.eps = eps
    
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

    def sigmoid_activation(self, alpha):
        s = 20.0
        x0 = 0.5
        return 1.0 / (1.0 + np.exp(-s*(alpha - x0)))
        # ReLU
        # alpha0 = 0.5
        # return max(0, (alpha - alpha0) / (1 - alpha0))

    def get_epsilon_nonlinear(self, velocity_field, node_patches, h_CG, u_n, stiffness_matrix, plot_func):
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
                delta_u = u_n.x.array[adj_node] - u_n.x.array[node]
                beta = stiffness_matrix.getValue(node, adj_node)

                numerator += beta * (delta_u)
                denominator += np.abs(beta) * np.abs(delta_u)

            numerator = np.abs(numerator)
            denominator = max(denominator, self.eps)
            alpha = numerator / denominator
            psi_alpha = self.sigmoid_activation(alpha)
            # print(alpha, ":", psi_alpha)
            plot_func.x.array[node] = psi_alpha
            epsilon.x.array[node] = psi_alpha * self.Cm * hi * fi_norm
        
        return epsilon

    def get_epsilon_linear(self, w, node_patches, h_CG, u_n, stiffness_matrix, plot_func):
        V = fem.functionspace(self.domain, ("Lagrange", 1))
        epsilon = fem.Function(V)
        for node, adjacent_nodes in node_patches.items():
            # node = i and adjacent_nodes (including self) = j
            hi = h_CG.x.array[node]
            w_values = w.x.array.reshape((-1, self.domain.geometry.dim))
            fi = w_values[node]
            fi_norm = np.linalg.norm(fi)

            numerator = 0
            denominator = 0
            for adj_node in adjacent_nodes:
                delta_u = u_n.x.array[adj_node] - u_n.x.array[node]
                beta = stiffness_matrix.getValue(node, adj_node)

                numerator += beta * delta_u
                denominator += np.abs(beta) * np.abs(delta_u)

            numerator = np.abs(numerator)
            denominator = max(denominator, self.eps)
            alpha = numerator / denominator
            alpha_psi = self.sigmoid_activation(alpha)
            plot_func.x.array[node] = alpha_psi
            epsilon.x.array[node] = alpha_psi * self.Cm * hi * fi_norm
        
        return epsilon
    
    """ Alternative: Iterate through stiffness matrix instead of patches """
    # def get_epsilon_nonlinear(self, velocity_field, node_patches, h_CG, u_n, stiffness_matrix, plot_func):
    #     # Create a function to store epsilon
    #     V = fem.functionspace(self.domain, ("Lagrange", 1))
    #     epsilon = fem.Function(V)
    #     epsilon_array = epsilon.x.petsc_vec  # Direct array access for efficiency

    #     # Access stiffness matrix as PETSc Mat
    #     mat = stiffness_matrix
    #     start, end = mat.getOwnershipRange()

    #     # Extract array data
    #     u_values = u_n.x.array
    #     # w_values = w.x.array.reshape((-1, self.domain.geometry.dim))
    #     h_values = h_CG.x.array

    #     # Loop through rows corresponding to local nodes
    #     for row in range(start, end):
    #         cols, bij = mat.getRow(row)  # Retrieve non-zero entries
    #         node = row
    #         hi = h_values[node]
    #         w = velocity_field(u_n.x.array[node])
    #         fi = np.array(w, dtype = 'float')
    #         fi_norm = np.linalg.norm(fi)

    #         numerator = 0.0
    #         denominator = 0.0

    #         nonzero_mask = bij != 0
    #         cols = cols[nonzero_mask]
    #         bij = bij[nonzero_mask]

    #         for idx, adj_node in enumerate(cols):
    #             if node != adj_node:  # Ignore diagonal entries
    #                 delta_u = u_values[adj_node] - u_values[node]
    #                 beta = bij[idx]

    #                 numerator += beta * delta_u
    #                 denominator += abs(beta) * abs(delta_u)

    #         numerator = abs(numerator)
    #         denominator = max(denominator, self.eps)
    #         alpha = numerator / denominator
    #         alpha_psi = self.sigmoid_activation(alpha)
    #         plot_func.x.array[node] = alpha_psi

    #         # Store computed epsilon
    #         epsilon_array[node] = alpha_psi * self.Cm * hi * fi_norm

    #     return epsilon
    # def get_epsilon_linear(self, w, node_patches, h_CG, u_n, stiffness_matrix, numerator_func):
    #     # Create a function to store epsilon
    #     V = fem.functionspace(self.domain, ("Lagrange", 1))
    #     epsilon = fem.Function(V)
    #     epsilon_array = epsilon.x.petsc_vec  # Direct array access for efficiency

    #     # Access stiffness matrix as PETSc Mat
    #     mat = stiffness_matrix
    #     start, end = mat.getOwnershipRange()

    #     # Extract array data
    #     u_values = u_n.x.array
    #     w_values = w.x.array.reshape((-1, self.domain.geometry.dim))
    #     h_values = h_CG.x.array

    #     # Loop through rows corresponding to local nodes
    #     for row in range(start, end):
    #         cols, bij = mat.getRow(row)  # Retrieve non-zero entries
    #         node = row
    #         hi = h_values[node]
    #         fi = w_values[node]
    #         fi_norm = np.linalg.norm(fi)

    #         numerator = 0.0
    #         denominator = 0.0

    #         nonzero_mask = bij != 0
    #         cols = cols[nonzero_mask]
    #         bij = bij[nonzero_mask]

    #         for idx, adj_node in enumerate(cols):
    #             if node != adj_node:  # Ignore diagonal entries
    #                 delta_u = u_values[adj_node] - u_values[node]
    #                 beta = bij[idx]

    #                 numerator += beta * delta_u
    #                 denominator += abs(beta) * abs(delta_u)

    #         numerator = abs(numerator)
    #         denominator = max(denominator, 1e-8)
    #         alpha = numerator / denominator

    #         # Store computed epsilon
    #         epsilon_array[node] = alpha * self.Cm * hi * fi_norm

    #     return epsilon