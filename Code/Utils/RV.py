import ufl
import numpy as np

from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
class RV:
    def __init__(self, Cvel, Crv, domain):
        self.Cvel = Cvel
        self.Crv = Crv
        self.domain = domain

    def get_epsilon(self, uh, velocity_field, residual, h):
        V = fem.functionspace(self.domain, ("Lagrange", 1))
        epsilon = fem.Function(V)
        
        # TODO: use num_nodes instead of residual.x.array.size
        for node in range(residual.x.array.size):
            hi = h.x.array[node]
            Ri = residual.x.array[node]
            w = velocity_field(uh.x.array[node])
            fi = np.array(w, dtype = 'float')
            fi_norm = np.linalg.norm(fi)
            epsilon.x.array[node] = min(self.Cvel * hi * fi_norm, self.Crv * hi ** 2 * np.abs(Ri))
        
        return epsilon
    
    def get_epsilon_1storder(self, uh, velocity_field, residual, h):
        V = fem.functionspace(self.domain, ("Lagrange", 1))
        epsilon = fem.Function(V)
        
        # TODO: use num_nodes instead of residual.x.array.size
        for node in range(residual.x.array.size):
            hi = h.x.array[node]
            w = velocity_field(uh.x.array[node])
            fi = np.array(w, dtype = 'float')
            fi_norm = np.linalg.norm(fi)

            epsilon.x.array[node] = 0.5*hi*fi_norm        
        return epsilon

    def get_epsilon_nonlinear(self, uh, u_n, velocity_field, Rh, h_CG, node_patches): #From Murtazos notes and paper, good one
        V = fem.functionspace(self.domain, ("Lagrange", 1))
        epsilon = fem.Function(V)
        absolute_term = np.linalg.norm(uh.x.array - np.mean(uh.x.array), ord=np.inf)

        for node, adjacent_nodes in node_patches.items():
            # node = i and adjacent_nodes (including self) = j
            # print("Node:", node, " - Adjacent nodes:", adjacent_nodes)

            #if node in boundary_nodes:
            #    Rh_normalized[node] = 0
            #else:
            u_i = np.zeros(len(adjacent_nodes))
            Rh_patch = np.zeros(len(adjacent_nodes))
            beta_patch = np.zeros(len(adjacent_nodes))
            j = 0
            for adj_node in adjacent_nodes:
                u_i[j] = u_n.x.array[adj_node]

                Rh_patch[j] = np.abs(Rh.x.array[adj_node])

                w = velocity_field(uh.x.array[adj_node])
                fi = np.array(w, dtype = 'float')
                fi_norm = np.linalg.norm(fi)
                beta_patch[j] = fi_norm
                j += 1
            u_tilde = np.max(u_i) - np.min(u_i)
            n_i = np.abs(u_tilde - absolute_term)
            Rh_i = np.max(Rh_patch)
            Ri = Rh_i / n_i
            beta_norm = np.max(beta_patch)
            hi = h_CG.x.array[node]
            epsilon.x.array[node] = min(self.Cvel * hi * beta_norm, self.Crv * hi ** 2 * np.abs(Ri))

        return epsilon
    
    def get_epsilon_linear(self, uh, u_n, velocity_field, Rh, h_CG, node_patches): #From Murtazos notes and paper, good one
        V = fem.functionspace(self.domain, ("Lagrange", 1))
        epsilon = fem.Function(V)
        absolute_term = np.linalg.norm(uh.x.array - np.mean(uh.x.array), ord=np.inf)

        for node, adjacent_nodes in node_patches.items():
            # node = i and adjacent_nodes (including self) = j
            # print("Node:", node, " - Adjacent nodes:", adjacent_nodes)
            #hi = h_CG.x.array[node]

            #if node in boundary_nodes:
            #    Rh_normalized[node] = 0
            #else:
            u_i = np.zeros(len(adjacent_nodes))
            Rh_patch = np.zeros(len(adjacent_nodes))
            beta_patch = np.zeros(len(adjacent_nodes))
            j = 0
            for adj_node in adjacent_nodes:
                u_i[j] = u_n.x.array[adj_node]

                Rh_patch[j] = np.abs(Rh.x.array[adj_node])
                w_values = velocity_field.x.array.reshape((-1, self.domain.geometry.dim))
                fi = w_values[node]
                fi_norm = np.linalg.norm(fi)
                beta_patch[j] = fi_norm
                j += 1

            u_tilde = np.max(u_i) - np.min(u_i)
            n_i = np.abs(u_tilde - absolute_term)
            Rh_i = np.max(Rh_patch)
            Ri = Rh_i / n_i
            beta_norm = np.max(beta_patch)
            hi = h_CG.x.array[node]
            epsilon.x.array[node] = min(self.Cvel * hi * beta_norm, self.Crv * hi ** 2 * np.abs(Ri))

        return epsilon
    