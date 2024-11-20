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
            w = uh.x.array[node]
            w = velocity_field(uh.x.array[node])
            fi = np.array(w, dtype = 'float')
            fi_norm = np.linalg.norm(fi)
            epsilon.x.array[node] = min(self.Cvel * hi * fi_norm, self.Crv * hi ** 2 * np.abs(Ri))