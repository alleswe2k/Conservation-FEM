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
    
    def normalize_Rh(self, uh, Rh, node_patches):
        #print(uh.x.array - np.mean(uh.x.array))
        #absolute_term = np.max(uh.x.array - np.mean(uh.x.array))
        absolute_term = np.linalg.norm(uh.x.array - np.mean(uh.x.array), ord=np.inf)
        #print(absolute_term)
        Rh_normalized = np.zeros(uh.x.array.size)
        for node in range(uh.x.array.size):
            u_i = np.zeros(uh.x.array.size)
            j = 0
            for i in node_patches[node]:
                u_i[j] = uh.x.array[i]
                j += 1
            u_tilde = np.max(u_i) - np.min(u_i)
            n_i = np.abs(u_tilde - absolute_term)
            Rh_normalized[node] = np.abs(Rh.x.array[node])/n_i
        return Rh_normalized

            
            


