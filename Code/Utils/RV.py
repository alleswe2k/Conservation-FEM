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

            u_i = np.zeros(len(adjacent_nodes)-1)
            #u_i[node] = uh.x.array[node]
            j = 0
            for adj_node in adjacent_nodes:
                if adj_node != node:
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

            u_i = np.zeros(len(adjacent_nodes)-1)
            #u_i[node] = uh.x.array[node]
            j = 0
            for adj_node in adjacent_nodes:
                if adj_node != node:
                    u_i[j] = uh.x.array[adj_node]
                    j += 1
            u_tilde = np.max(u_i) - np.min(u_i)
            n_i = np.abs(u_tilde - absolute_term)
            ni[node] = n_i
       
        return ni
    
    def find_ni_robust(self, uh, Rh, node_patches): #Robust

            absolute_term = np.linalg.norm(uh.x.array - np.mean(uh.x.array), ord=np.inf)
            ni = np.zeros(Rh.x.array.size)

            for node, adjacent_nodes in node_patches.items():
                # node = i and adjacent_nodes (including self) = j
                # print("Node:", node, " - Adjacent nodes:", adjacent_nodes)

                u_i = np.zeros(len(adjacent_nodes))
                j = 0
                for adj_node in adjacent_nodes:
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

            u_i = np.zeros(len(adjacent_nodes)-1)
            Rh_patch = np.zeros(len(adjacent_nodes))
            #input()
            #u_i[node] = uh.x.array[node]
            j = 0
            i = 0
            for adj_node in adjacent_nodes:
                #i += 1
                if adj_node != node:
                    u_i[i] = uh.x.array[adj_node]
                    i+=1
                Rh_patch[j] = np.abs(Rh.x.array[adj_node])
                j += 1
                #print(node)
                #print(adj_node)
                #input()
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

            Rh_patch = np.zeros(len(adjacent_nodes))
            j = 0
            for adj_node in adjacent_nodes:
                Rh_patch[j] = np.abs(Rh.x.array[adj_node])
                j += 1
            Rh_i = np.max(Rh_patch)
            Rh_normalized[node] = Rh_i/n
       
        return Rh_normalized
    
    
    def normalize_Rh_robust_x2(self, uh, Rh, node_patches): #From Murtazos notes and paper

        absolute_term = np.linalg.norm(uh.x.array - np.mean(uh.x.array), ord=np.inf)
        Rh_normalized = np.zeros(Rh.x.array.size)

        for node, adjacent_nodes in node_patches.items():
            # node = i and adjacent_nodes (including self) = j
            # print("Node:", node, " - Adjacent nodes:", adjacent_nodes)
            #hi = h_CG.x.array[node]

            #if node in boundary_nodes:
            #    Rh_normalized[node] = 0
            #else:
            u_i = np.zeros(len(adjacent_nodes))
            Rh_patch = np.zeros(len(adjacent_nodes))
            j = 0
            for adj_node in adjacent_nodes:
                u_i[j] = uh.x.array[adj_node]
                Rh_patch[j] = np.abs(Rh.x.array[adj_node])
                j += 1
            u_tilde = np.max(u_i) - np.min(u_i)
            n_i = np.abs(u_tilde - absolute_term)
            Rh_i = np.max(Rh_patch)
            Rh_normalized[node] = Rh_i/n_i
          
        return Rh_normalized
    
    def normalize_Rh_robust_2(self, uh, Rh, node_patches): #From Murtazos notes

        absolute_term = np.linalg.norm(uh.x.array - np.mean(uh.x.array), ord=np.inf)
        Rh_normalized = np.zeros(Rh.x.array.size)

        for node, adjacent_nodes in node_patches.items():
            # node = i and adjacent_nodes (including self) = j
            # print("Node:", node, " - Adjacent nodes:", adjacent_nodes)

            #u_i = np.zeros(len(adjacent_nodes))
            #Rh_patch = np.zeros(len(adjacent_nodes))

            normalized_patch = np.zeros(len(adjacent_nodes))

            j = 0
            for adj_node in adjacent_nodes:

                Rh_j = np.abs(Rh.x.array[adj_node])

                normalized_patch[j] = Rh_j / absolute_term

                j += 1

            Rh_normalized[node] = np.max(normalized_patch)
          
        return Rh_normalized


    def get_epsilon_x3(self, uh, u_n, velocity_field, Rh, h, node_patches): #From Murtazos notes and paper
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

                Rh_patch[j] = np.abs(Rh.x.array[adj_node]) / absolute_term

                w = velocity_field(uh.x.array[adj_node])
                fi = np.array(w, dtype = 'float')
                fi_norm = np.linalg.norm(fi)
                beta_patch[j] = fi_norm
                j += 1

            Ri = np.max(Rh_patch)
            beta_norm = np.max(beta_patch)
            hi = h.x.array[node]
            epsilon.x.array[node] = min(self.Cvel * hi * beta_norm, self.Crv * hi ** 2 * np.abs(Ri))

        return epsilon


    
    def get_epsilon_robust(self, uh, velocity_field, residual, h, node_patches):
        V = fem.functionspace(self.domain, ("Lagrange", 1))
        epsilon = fem.Function(V)
        
        # TODO: use num_nodes instead of residual.x.array.size
        for node, adjacent_nodes in node_patches.items():

            beta_patch = np.zeros(len(adjacent_nodes))

            j = 0
            for adj_node in adjacent_nodes:

                w = velocity_field(uh.x.array[adj_node])
                fi = np.array(w, dtype = 'float')
                fi_norm = np.linalg.norm(fi)
                beta_patch[j] = fi_norm

                j += 1
            
            beta_norm = np.max(beta_patch)

            hi = h.x.array[node]
            Ri = residual.x.array[node]
            epsilon.x.array[node] = min(self.Cvel * hi * beta_norm, self.Crv * hi ** 2 * np.abs(Ri))
        
        return epsilon
    
    def get_epsilon_x2(self, uh, u_n, velocity_field, Rh, h, node_patches): #From Murtazos notes and paper
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
            hi = h.x.array[node]
            epsilon.x.array[node] = min(self.Cvel * hi * beta_norm, self.Crv * hi ** 2 * np.abs(Ri))

        return epsilon