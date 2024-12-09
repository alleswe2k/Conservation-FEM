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

fraction = 32
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

# def reflecting_bc(x):
#     n = fem.Function(V)
#     n_x = x[0] / np.sqrt(x[0] ** 2 + x[1] ** 2)
#     n_y = x[1] / np.sqrt(x[0] ** 2 + x[1] ** 2)
#     return np.array([n_x, n_y])

# reflecting_bc = ufl.FacetNormal(domain)

# def reflect_velocity(u, n):
#     return u - 2 * ufl.dot(u, n) * n


V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, ))) # Vector valued function space
Q = fem.functionspace(domain, ("Lagrange", 1)) # Scalar valued function space

def walls(x):
    return np.logical_or(np.isclose(x[1], -0.3), np.isclose(x[1], 0.3))

wall_dofs = fem.locate_dofs_geometrical(V, walls)
u_noslip = np.array((0,) * domain.geometry.dim, dtype=PETSc.ScalarType)
bc_noslip = fem.dirichletbc(u_noslip, wall_dofs, V)

rho_n = fem.Function(Q)
m_n = fem.Function(V)
e_n = fem.Function(Q)

rho_n.interpolate(init_density)
m_n.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))
e_n.interpolate(lambda x: np.ones(x.shape[1]))

gamma = 1.4


u_n = fem.Function(V)
u_n.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))

p_n = fem.Function(Q)
p_n.interpolate(init_pressure)

T_n = fem.Function(Q)
T_n.interpolate(lambda x: np.ones(x.shape[1]))

number_of_nodes = p_n.x.array.size

def calc_physical_quantities():
    m_vals = m_n.x.array.reshape((-1, domain.geometry.dim))
    u_vals = np.zeros_like(m_vals)

    for node in range(number_of_nodes):
        rho_i = rho_n.x.array[node]
        e_i = e_n.x.array[node]
        m_i = m_vals[node]

        u_vals[node] = m_i / rho_i
        
        T_i = e_i / rho_i - np.dot(u_vals[node], u_vals[node]) / 2
        p_n.x.array[node] = (gamma - 1) * rho_i * T_i

        T_n.x.array[node] = T_i
    
    u_n.x.array[:] = u_vals.flatten()

def cfl_cond():
    u_vals = u_n.x.array.reshape((-1, domain.geometry.dim))
    lst = [1]
    for node in range(number_of_nodes):
        u_len = np.linalg.norm(u_vals[node])
        if u_len != 0:
            lst.append(hmax / u_len)
    return min(np.min(lst), 1e-1)

dt = cfl_cond()

rho_h = fem.Function(Q)
rho_h.name = 'rho_h'
rho_h.x.array[:] = rho_n.x.array

m_h = fem.Function(V)
m_h.name = 'm_h'
m_h.x.array[:] = m_n.x.array

e_h = fem.Function(Q)
e_h.name = 'e_h'
e_h.x.array[:] = e_n.x.array

rho_trial, rho_test = ufl.TrialFunction(Q), ufl.TestFunction(Q)
m_trial, m_test = ufl.TrialFunction(V), ufl.TestFunction(V)
e_trial, e_test = ufl.TrialFunction(Q), ufl.TestFunction(Q)
 

# Formulation for density
F1 = (rho_trial * rho_test * ufl.dx -
    0.5 * dt * rho_trial * ufl.dot(u_n, ufl.grad(rho_test)) * ufl.dx -
    rho_n * rho_test * ufl.dx -
    0.5 * dt * rho_n * ufl.dot(u_n, ufl.grad(rho_test)) * ufl.dx)
a1 = fem.form(ufl.lhs(F1))
L1 = fem.form(ufl.rhs(F1))

A1 = assemble_matrix(a1, bcs=[bc_noslip])
A1.assemble()
b1 = create_vector(L1)

# Solver for density
solver1 = PETSc.KSP().create(domain.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.BCGS)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.HYPRE)
pc1.setHYPREType("boomeramg")

T = 1
t = 0

plot_grid()

with b1.localForm() as loc_1:
    loc_1.set(0)
assemble_vector(b1, L1)
apply_lifting(b1, [a1], [[bc_noslip]])
b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
set_bc(b1, [bc_noslip])
solver1.solve(b1, rho_h.x.petsc_vec)
rho_h.x.scatter_forward()

print(rho_h.x.array)

pde.plot_pv_2d(domain, fraction, rho_h, 'After one iteration', 'euler', location_figures)

# while t < T:
#     t += dt

#     # Solving for density
#     with b1.localForm() as loc_1:
#         loc_1.set(0)
#     assemble_vector(b1, L1)
#     apply_lifting(b1, [a1], [[bc_noslip]])
#     b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
#     set_bc(b1, [bc_noslip])
#     solver1.solve(b1, rho_h.x.petsc_vec)
#     rho_h.x.scatter_forward()


#     rho_n.x.array[:] = rho_h.x.array

#     calc_physical_quantities()
#     dt = cfl_cond()
    


