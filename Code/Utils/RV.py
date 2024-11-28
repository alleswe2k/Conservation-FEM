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
    

    
    def normalize_Rh(self, uh, Rh, node_patches): #From the paper

        absolute_term = np.linalg.norm(uh.x.array - np.mean(uh.x.array), ord=np.inf)
        Rh_normalized = np.zeros(Rh.x.array.size)

        for node, adjacent_nodes in node_patches.items():
            # node = i and adjacent_nodes (including self) = j
            # print("Node:", node, " - Adjacent nodes:", adjacent_nodes)
            #hi = h_CG.x.array[node]

            u_i = np.zeros(len(adjacent_nodes))
            #input()
            #u_i[node] = uh.x.array[node]
            j = 0
            for adj_node in adjacent_nodes:
                #i += 1
                u_i[j] = uh.x.array[adj_node]
                j += 1
            u_tilde = np.max(u_i) - np.min(u_i)
            n_i = np.abs(u_tilde - absolute_term)
            Rh_normalized[node] = np.abs(Rh.x.array[node])/n_i
       
        return Rh_normalized
    

    def find_ni(self, uh, Rh, node_patches): #From paper

        absolute_term = np.linalg.norm(uh.x.array - np.mean(uh.x.array), ord=np.inf)
        ni = np.zeros(Rh.x.array.size)

        for node, adjacent_nodes in node_patches.items():
            # node = i and adjacent_nodes (including self) = j
            # print("Node:", node, " - Adjacent nodes:", adjacent_nodes)
            #hi = h_CG.x.array[node]

            u_i = np.zeros(len(adjacent_nodes))
            Rh_patch = np.zeros(len(adjacent_nodes)+1)
            #input()
            #u_i[node] = uh.x.array[node]
            j = 0
            for adj_node in adjacent_nodes:
                #i += 1
                u_i[j] = uh.x.array[adj_node]
                j += 1
            u_tilde = np.max(u_i) - np.min(u_i)
            n_i = np.abs(u_tilde - absolute_term)
            ni[node] = n_i
       
        return ni
    
    def normalize_Rh_robust_ni(self, uh, Rh, node_patches): #From Murtazos notes and paper

        absolute_term = np.linalg.norm(uh.x.array - np.mean(uh.x.array), ord=np.inf)
        Rh_normalized = np.zeros(Rh.x.array.size)

        for node, adjacent_nodes in node_patches.items():
            # node = i and adjacent_nodes (including self) = j
            # print("Node:", node, " - Adjacent nodes:", adjacent_nodes)
            #hi = h_CG.x.array[node]

            u_i = np.zeros(len(adjacent_nodes))
            Rh_patch = np.zeros(len(adjacent_nodes)+1)
            Rh_patch[0] = Rh.x.array[node]
            #input()
            #u_i[node] = uh.x.array[node]
            j = 0
            for adj_node in adjacent_nodes:
                #i += 1
                u_i[j] = uh.x.array[adj_node]
                Rh_patch[j+1] = np.abs(Rh.x.array[adj_node])
                j += 1
            u_tilde = np.max(u_i) - np.min(u_i)
            n_i = np.abs(u_tilde - absolute_term)
            Rh_i = np.max(Rh_patch)
            Rh_normalized[node] = Rh_i/n_i
       
        return Rh_normalized
    
    def normalize_Rh_robust_simple(self, uh, Rh, node_patches): #From Murtazos notes

        n = np.linalg.norm(uh.x.array - np.mean(uh.x.array), ord=np.inf)
        Rh_normalized = np.zeros(Rh.x.array.size)

        for node, adjacent_nodes in node_patches.items():
            # node = i and adjacent_nodes (including self) = j
            # print("Node:", node, " - Adjacent nodes:", adjacent_nodes)
            #hi = h_CG.x.array[node]

            Rh_patch = np.zeros(len(adjacent_nodes)+1)
            Rh_patch[0] = Rh.x.array[node]
            #input()
            #u_i[node] = uh.x.array[node]
            j = 0
            for adj_node in adjacent_nodes:
                Rh_patch[j+1] = np.abs(Rh.x.array[adj_node])
                j += 1
            Rh_i = np.max(Rh_patch)
            Rh_normalized[node] = Rh_i/n
       
        return Rh_normalized

            
            


