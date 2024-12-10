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

PLOT = True
pde = PDE_plot()

max_iterations = 10
tolerance = 1e-2

gmsh.initialize()

membrane = gmsh.model.occ.addRectangle(-0.3,-0.3,0,0.6,0.6)
gmsh.model.occ.synchronize()

gdim = 2
gmsh.model.addPhysicalGroup(gdim, [membrane], 1)

fraction = 128
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

V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, ))) # Vector valued function space
Q = fem.functionspace(domain, ("Lagrange", 1)) # Scalar valued function space

def walls(x):
    return np.logical_or.reduce((
        np.isclose(x[1], -0.3),  # Bottom wall
        np.isclose(x[1], 0.3),   # Top wall
        np.isclose(x[0], -0.3),  # Left wall
        np.isclose(x[0], 0.3)    # Right wall
    ))

wall_dofs = fem.locate_dofs_geometrical(Q, walls)
bc_noslip = fem.dirichletbc(PETSc.ScalarType(0), wall_dofs, Q)

rho_n = fem.Function(Q)
mx_n = fem.Function(Q)
my_n = fem.Function(Q)
e_n = fem.Function(Q)

rho_n.interpolate(init_density)
mx_n.interpolate(lambda x: np.zeros(x.shape[1]))
my_n.interpolate(lambda x: np.zeros(x.shape[1]))

gamma = 1.4
temperature = 25 # degree celsius

u_n = fem.Function(V)
u_n.interpolate(lambda x: np.zeros((domain.geometry.dim, x.shape[1])))

p_n = fem.Function(Q)
p_n.interpolate(init_pressure)

T_n = fem.Function(Q)
T_n.interpolate(lambda x: np.ones(x.shape[1]) * temperature)

number_of_nodes = p_n.x.array.size

for node in range(number_of_nodes):
    e_n.x.array[node] = T_n.x.array[node] * rho_n.x.array[node]


def apply_slip_bc():
    for dof in wall_dofs:
        vector = u_n.x.array[dof * domain.geometry.dim : (dof + 1) * domain.geometry.dim]
        u_n.x.array[dof * domain.geometry.dim : (dof + 1) * domain.geometry.dim] = vector * -1

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

mx_h = fem.Function(Q)
mx_h.name = 'mx_h'
mx_h.x.array[:] = mx_n.x.array

my_h = fem.Function(Q)
my_h.name = 'my_h'
my_h.x.array[:] = my_n.x.array

e_h = fem.Function(Q)
e_h.name = 'e_h'
e_h.x.array[:] = e_n.x.array

def calc_physical_quantities():
    u_vals = u_n.x.array.reshape((-1, domain.geometry.dim))

    for node in range(number_of_nodes):
        rho_i = 0.5 * (rho_n.x.array[node] + rho_h.x.array[node])
        e_i = 0.5 * (e_n.x.array[node] + e_h.x.array[node])
        mx_i = 0.5 * (mx_n.x.array[node] + mx_h.x.array[node])
        my_i = 0.5 * (my_n.x.array[node] + my_h.x.array[node])
        u_vals[node][0] = mx_i / rho_i
        u_vals[node][1] = my_i / rho_i
        
        T_i = e_i / rho_i - np.dot(u_vals[node], u_vals[node]) / 2
        p_n.x.array[node] = (gamma - 1) * rho_i * T_i

        T_n.x.array[node] = T_i
    
    u_n.x.array[:] = u_vals.flatten()

rho_trial, rho_test = ufl.TrialFunction(Q), ufl.TestFunction(Q)
mx_trial, mx_test = ufl.TrialFunction(Q), ufl.TestFunction(Q)
my_trial, my_test = ufl.TrialFunction(Q), ufl.TestFunction(Q)
e_trial, e_test = ufl.TrialFunction(Q), ufl.TestFunction(Q)


# Formulation for density
F1 = (rho_trial * rho_test * ufl.dx -
    0.5 * dt * rho_trial * ufl.dot(u_n, ufl.grad(rho_test)) * ufl.dx -
    rho_n * rho_test * ufl.dx -
    0.5 * dt * rho_n * ufl.dot(u_n, ufl.grad(rho_test)) * ufl.dx)
a1 = fem.form(ufl.lhs(F1))
L1 = fem.form(ufl.rhs(F1))

A1 = assemble_matrix(a1)
A1.assemble()
b1 = create_vector(L1)

# Formulation for momentum in x
F2 = mx_trial * mx_test * ufl.dx - 0.5 * dt * mx_trial * ufl.dot(u_n, ufl.grad(mx_test)) * ufl.dx - dt * p_n * ufl.grad(mx_test)[0] * ufl.dx
F2 -= mx_n * mx_test * ufl.dx + 0.5 * dt * mx_n * ufl.dot(u_n, ufl.grad(mx_test)) * ufl.dx
a2 = fem.form(ufl.lhs(F2))
L2 = fem.form(ufl.rhs(F2))

A2 = assemble_matrix(a2)
A2.assemble()
b2 = create_vector(L2)

# Formulation for momentum in y
F3 = ufl.dot(my_trial, my_test) * ufl.dx - 0.5 * dt * my_trial * ufl.dot(u_n, ufl.grad(my_test)) * ufl.dx - dt * p_n * ufl.grad(my_test)[1] * ufl.dx
F3 -= ufl.dot(my_n, my_test) * ufl.dx + 0.5 * dt * my_n * ufl.dot(u_n, ufl.grad(my_test)) * ufl.dx
a3 = fem.form(ufl.lhs(F3))
L3 = fem.form(ufl.rhs(F3))

A3 = assemble_matrix(a3)
A3.assemble()
b3 = create_vector(L3)


# Formulation for energy
F4 = e_trial * e_test * ufl.dx - 0.5 * dt * ufl.dot(u_n, ufl.grad(e_test)) * e_trial * ufl.dx + dt * ufl.div(ufl.outer(u_n, p_n)) * e_test * ufl.dx
F4 -= e_n * e_test * ufl.dx + 0.5 * dt * ufl.dot(u_n, ufl.grad(e_test)) * e_n * ufl.dx
a4 = fem.form(ufl.lhs(F3))
L4 = fem.form(ufl.rhs(F3))

A4 = assemble_matrix(a4)
A4.assemble()
b4 = create_vector(L4)

# Solver for density
solver1 = PETSc.KSP().create(domain.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.PREONLY)
solver1.getPC().setType(PETSc.PC.Type.LU)

# Solver for momentum in x
solver2 = PETSc.KSP().create(domain.comm)
solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.PREONLY)
solver2.getPC().setType(PETSc.PC.Type.LU)

# Solver for momentum in y
solver3 = PETSc.KSP().create(domain.comm)
solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.PREONLY)
solver3.getPC().setType(PETSc.PC.Type.LU)

# Solver for energy
solver4 = PETSc.KSP().create(domain.comm)
solver4.setOperators(A4)
solver4.setType(PETSc.KSP.Type.PREONLY)
solver4.getPC().setType(PETSc.PC.Type.LU)

if PLOT:
    # pyvista.start_xvfb()

    grid = pv.UnstructuredGrid(*plot.vtk_mesh(V))

    plotter = pv.Plotter()
    plotter.open_gif("euler.gif", fps=1)

    grid.point_data["rho_h"] = rho_h.x.array
    warped = grid.warp_by_scalar("rho_h", factor=1)

    viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
    sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                position_x=0.1, position_y=0.8, width=0.8, height=0.1)

    renderer = plotter.add_mesh(warped, show_edges=False, lighting=False,
                                cmap=viridis, scalar_bar_args=sargs,
                                clim=[min(rho_h.x.array), max(rho_h.x.array)])
    plotter.view_xy()


T = 1
t = 0
i = 0

while t < T:

    i += 1
    t += 12
    # Solve for density
    with b1.localForm() as loc_1:
        loc_1.set(0)
    assemble_vector(b1, L1)
    # apply_lifting(b1, [a1], [[bc_noslip]])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    # set_bc(b1, [bc_noslip])
    solver1.solve(b1, rho_h.x.petsc_vec)
    rho_h.x.scatter_forward()

    with b2.localForm() as loc_2:
        loc_2.set(0)
    assemble_vector(b2, L2)
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver2.solve(b2, mx_h.x.petsc_vec)
    mx_h.x.scatter_forward()

    with b3.localForm() as loc_3:
        loc_3.set(0)
    assemble_vector(b3, L3)
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver3.solve(b3, my_h.x.petsc_vec)
    my_h.x.scatter_forward()

    with b4.localForm() as loc_4:
        loc_4.set(0)
    assemble_vector(b4, L4)
    b4.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver4.solve(b4, e_h.x.petsc_vec)
    e_h.x.scatter_forward()

    calc_physical_quantities()

    for dof in wall_dofs:
        vector = u_n.x.array[dof * domain.geometry.dim : (dof + 1) * domain.geometry.dim]
        u_n.x.array[dof * domain.geometry.dim : (dof + 1) * domain.geometry.dim] = vector * -1

        mx_h.x.array[dof] *= -1
        my_h.x.array[dof] *= -1

    rho_n.x.array[:] = rho_h.x.array
    mx_n.x.array[:] = mx_h.x.array
    mx_n.x.array[:] = mx_h.x.array
    e_n.x.array[:] = e_h.x.array

    print(f"t = {t}, dt = {dt}")
    dt = cfl_cond()
    # t = 10
    if PLOT:
        new_warped = grid.warp_by_scalar("rho_h", factor=1)
        warped.points[:, :] = new_warped.points
        warped.point_data["rho_h"][:] = rho_h.x.array
        plotter.write_frame()

if PLOT:
    plotter.close()


pde.plot_pv_2d(domain, fraction, rho_h, f'After {i} iterations', 'rho', location_figures)
pde.plot_pv_2d(domain, fraction, e_h, f'After {i} iterations', 'energy', location_figures)
