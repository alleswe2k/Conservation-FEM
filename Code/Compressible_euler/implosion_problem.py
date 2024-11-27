import pyvista as pv

import ufl
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

V = fem.functionspace(domain, ("Lagrange", 1))
DG0 = fem.functionspace(domain, ("DG", 0))

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

p_n = fem.Function(V)
p_n.name = "p_n"
p_n.interpolate(init_pressure)

pde.plot_pv_2d(domain, fraction, p_n, 'Pressure', 'pressure', location_figures)

def reflecting_bc(x):
    n = fem.Function(V)
    n_x = x[0] / np.sqrt(x[0] ** 2 + x[1] ** 2)
    n_y = x[1] / np.sqrt(x[0] ** 2 + x[1] ** 2)
    return np.array([n_x, n_y])

def reflect_velocity(u, n):
    return u - 2 * ufl.dot(u, n) * n

u = fem.Function(V)
n = fem.Constant(domain, reflect_velocity(u, reflecting_bc))

dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], -0.3) | np.isclose(x[0], 0.3) | np.isclose(x[1], -0.3) | np.isclose(x[1], 0.3))

bc = fem.dirichletbc(n, dofs)


rho_0 = fem.Function(V)
m_0 = fem.Function(V)
e_0 = fem.Function(V)

rho_0.interpolate(init_density)
m_0.interpolate(lambda x: 0.0)
e_0.interpolate(lambda x: 1.0)


v = ufl.TestFunction(V)
gamma = 1.4

rho, m1, m2, e = u.split()

T = e / rho - 0.5 * (m1**2 + m2**2) / rho**2
p = (gamma - 1) * rho * T
