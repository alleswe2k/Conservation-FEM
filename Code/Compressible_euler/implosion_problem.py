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
from Utils.PDE_plot import PDE_plot

from Utils.helpers import get_nodal_h
from Utils.RV import RV

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
location_figures = os.path.join(script_dir, 'Figures/GFEM') # location = './Figures'
location_data = os.path.join(script_dir, 'Data') # location = './Data'

pde = PDE_plot()

gmsh.initialize()

membrane = gmsh.model.occ.addRectangle(-0.3,-0.3,0,0.6,0.6)
gmsh.model.occ.synchronize()

gdim = 2
gmsh.model.addPhysicalGroup(gdim, [membrane], 1)

fraction = 64
hmax = 1/fraction # 0.05 in example
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmax)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)
gmsh.model.mesh.generate(gdim)

gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

def plot_grid():
    tdim = domain.topology.dim
    os.environ["PYVISTA_OFF_SCREEN"] = "True"
    pv.start_xvfb()
    plotter = pv.Plotter(off_screen=True)
    domain.topology.create_connectivity(tdim, tdim)
    topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)

    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    plotter.screenshot(f"{location_figures}/grid.png")


def init_pressure(x):
    p_in = 0.14
    p_out = 1
    radius = 0.15
    inside = np.abs(x[0]) + np.abs(x[1]) <= radius
    return np.where(inside, p_in, p_out)
    
def init_density(x):
    p_in = 0.125
    p_out = 1
    radius = 0.15
    inside = np.abs(x[0]) + np.abs(x[1]) <= radius
    return np.where(inside, p_in, p_out)

def reflecting_bc(x):
    n = fem.Function(V)
    n_x = x[0] / np.sqrt(x[0] ** 2 + x[1] ** 2)
    n_y = x[1] / np.sqrt(x[0] ** 2 + x[1] ** 2)
    return np.array([n_x, n_y])

reflecting_bc = ufl.FacetNormal(domain)

def reflect_velocity(u, n):
    return u - 2 * ufl.dot(u, n) * n

v_cg2 = element("Lagrange", domain.topology.cell_name(), 2, shape=(domain.geometry.dim, ))
s_cg1 = element("Lagrange", domain.topology.cell_name(), 1)
V = fem.functionspace(domain, v_cg2) # Vector valued function space
Q = fem.functionspace(domain, s_cg1) # Scalar valued function space

def walls(x):
    return np.logical_or(np.isclose(x[1], -0.3), np.isclose(x[1], 0.3))

wall_dofs = fem.locate_dofs_geometrical(V, walls)
u_noslip = np.array((0,) * domain.geometry.dim, dtype=PETSc.ScalarType)
bc_noslip = fem.dirichletbc(u_noslip, wall_dofs, V)

rho_n = fem.Function(Q)
m_n = fem.Function(V)
e_n = fem.Function(Q)

rho_n.interpolate(init_density)
m_n.interpolate(lambda x: 0.0)
e_n.interpolate(lambda x: 1.0)
gamma = 1.4


u_n = fem.Function(V)
p_n = fem.Function(Q)
T_n = fem.Function(Q)

u_n = m_n / rho_n

print(u_n.x.array)




# def cfl_cond():
#     pass




# rho_trial, rho_test = ufl.TrialFunction(Q), ufl.TestFunction(Q)





