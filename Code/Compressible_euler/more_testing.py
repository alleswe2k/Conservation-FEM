import matplotlib as mpl
import pyvista as pv

import ufl
from basix.ufl import element
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

import gmsh
from dolfinx.io import gmshio

from dolfinx import fem, mesh, plot
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl.finiteelement
from Utils.PDE_plot import PDE_plot

from Utils.helpers import get_nodal_h
from Utils.RV import RV

from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
location_figures = os.path.join(script_dir, 'Figures/GFEM') # location = './Figures'
location_data = os.path.join(script_dir, 'Data') # location = './Data'

PLOT = False
pde = PDE_plot()

max_iterations = 10
tolerance = 1e-2

gmsh.initialize()

membrane = gmsh.model.occ.addRectangle(-0.3,-0.3,0,0.6,0.6)
gmsh.model.occ.synchronize()

gdim = 2
gmsh.model.addPhysicalGroup(gdim, [membrane], 1)

fraction = 2
hmax = 1/fraction # 0.05 in example
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmax)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)
gmsh.model.mesh.generate(gdim)

gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, ))) # Vector valued function space
Q = fem.functionspace(domain, ("Lagrange", 1)) # Scalar valued function space

def walls(x):
    return np.logical_or.reduce((
        np.isclose(x[1], -0.3),  # Bottom wall
        np.isclose(x[1], 0.3),   # Top wall
        np.isclose(x[0], -0.3),  # Left wall
        np.isclose(x[0], 0.3)    # Right wall
    ))

n = ufl.FacetNormal(domain)
wall_dofs = fem.locate_dofs_geometrical(V, walls)

u_n = fem.Function(V)
u_n.interpolate(lambda x: np.ones((domain.geometry.dim, x.shape[1])))

for dof in wall_dofs:
    vector = u_n.x.array[dof * domain.geometry.dim : (dof + 1) * domain.geometry.dim]
    u_n.x.array[dof * domain.geometry.dim : (dof + 1) * domain.geometry.dim] = vector * -1

print(u_n.x.array)



