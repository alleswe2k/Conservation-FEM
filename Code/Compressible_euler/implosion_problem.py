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

fraction = 4
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

# Formulation for momentum
F2 = ufl.dot(m_trial, m_test) * ufl.dx - 0.5 * dt * ufl.dot(ufl.dot(u_n, ufl.grad(m_test)), m_trial) * ufl.dx - dt * p_n * ufl.div(m_test) * ufl.dx
F2 -= ufl.dot(m_n, m_test) * ufl.dx + 0.5 * dt * ufl.dot(ufl.dot(u_n, ufl.grad(m_test)), m_n) * ufl.dx
a2 = fem.form(ufl.lhs(F2))
L2 = fem.form(ufl.rhs(F2))

A2 = assemble_matrix(a2, bcs=[bc_noslip])
A2.assemble()
b2 = create_vector(L2)

# Formulation for energy
F3 = e_trial * e_test * ufl.dx - 0.5 * dt * ufl.dot(u_n, ufl.grad(e_test)) * e_trial * ufl.dx + dt * ufl.div(ufl.outer(u_n, p_n)) * e_test * ufl.dx
F3 -= e_n * e_test * ufl.dx + 0.5 * dt * ufl.dot(u_n, ufl.grad(e_test)) * e_n * ufl.dx
a3 = fem.form(ufl.lhs(F3))
L3 = fem.form(ufl.rhs(F3))

A3 = assemble_matrix(a3, bcs=[bc_noslip])
A3.assemble()
b3 = create_vector(L3)

# Solver for density
solver1 = PETSc.KSP().create(domain.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.BCGS)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.HYPRE)
pc1.setHYPREType("boomeramg")

# Solver for momentum
solver2 = PETSc.KSP().create(domain.comm)
solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.BCGS)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")

# Solver for energy
solver3 = PETSc.KSP().create(domain.comm)
solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.CG)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.SOR)

T = 1
t = 0

if PLOT:
    # pyvista.start_xvfb()

    grid = pv.UnstructuredGrid(*plot.vtk_mesh(V))

    plotter = pv.Plotter()
    plotter.open_gif("euler.gif", fps=10)

    grid.point_data["rho_h"] = rho_h.x.array
    warped = grid.warp_by_scalar("rho_h", factor=1)

    viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
    sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                position_x=0.1, position_y=0.8, width=0.8, height=0.1)

    renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
                                cmap=viridis, scalar_bar_args=sargs,
                                clim=[min(rho_h.x.array), max(rho_h.x.array)])
    plotter.view_xy()

while t < T:

    dt = cfl_cond()
    t += dt

    # Solving for density
    with b1.localForm() as loc_1:
        loc_1.set(0)
    assemble_vector(b1, L1)
    apply_lifting(b1, [a1], [[bc_noslip]])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, [bc_noslip])
    solver1.solve(b1, rho_h.x.petsc_vec)
    rho_h.x.scatter_forward()

    # Solving for momentum
    with b2.localForm() as loc_2:
        loc_2.set(0)
    assemble_vector(b2, L2)
    apply_lifting(b2, [a2], [[bc_noslip]])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, [bc_noslip])
    solver2.solve(b2, m_h.x.petsc_vec)
    m_h.x.scatter_forward()

    # Solving for energy
    with b3.localForm() as loc_3:
        loc_3.set(0)
    assemble_vector(b3, L3)
    apply_lifting(b3, [a3], [[bc_noslip]])
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b3, [bc_noslip])
    solver3.solve(b3, e_h.x.petsc_vec)
    e_h.x.scatter_forward()    

    rho_n.x.array[:] = rho_h.x.array
    m_n.x.array[:] = m_h.x.array
    e_n.x.array[:] = e_h.x.array

    calc_physical_quantities()
    print("Rho: ", rho_n.x.array)
    print("M: ", m_n.x.array)
    print("Vel: ", u_n.x.array)

    if PLOT:
        new_warped = grid.warp_by_scalar("rho_h", factor=1)
        warped.points[:, :] = new_warped.points
        warped.point_data["rho_h"][:] = rho_h.x.array
        plotter.write_frame()
    



