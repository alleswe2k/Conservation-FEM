import matplotlib as mpl
import pyvista
import ufl
import numpy as np
from tqdm import tqdm

from petsc4py import PETSc
from mpi4py import MPI

import gmsh
from dolfinx.io import gmshio

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, LinearProblem

from Utils.PDE_plot import PDE_plot
from Utils.SI import SI
from Utils.RV import RV
from Utils.helpers import get_nodal_h

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
location_figures = os.path.join(script_dir, 'Figures/RV')
location_data = os.path.join(script_dir, 'Data/RV/')

# Enable or disable real-time plotting
pde = PDE_plot()
PLOT = False
degree = 3
# Creating mesh
gmsh.initialize()

membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
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

V = fem.functionspace(domain, ("Lagrange", degree))
# domain.geometry.dim = (2, )
W = fem.functionspace(domain, ("Lagrange", degree, (domain.geometry.dim, ))) # Lagrange 2 in documentation
DG0 = fem.functionspace(domain, ("DG", 0))

# def initial_condition(x, r0=0.25, x0_1=0.3, x0_2=0):
#     return 1/2*(1-np.tanh(((x[0]-x0_1)**2+(x[1]-x0_2)**2)/r0**2 - 1))
def initial_condition(x, r0=0.25, x0_1=0.3, x0_2=0):
    return (x[0] - x0_1)**2 + (x[1] - x0_2)**2 <= r0**2

def velocity_field(x):
    return np.array([-2*np.pi*x[1], 2*np.pi*x[0]])

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

u_old = fem.Function(V)
u_old.name = "u_old"
u_old.interpolate(initial_condition)

u_ex = fem.Function(V)
u_ex.interpolate(initial_condition)

# velocity field f_prim
w = fem.Function(W)
w.name = "w"
w.interpolate(velocity_field)

epsilon = fem.Function(V)
epsilon.name = "epsilon"

w_values = w.x.array.reshape((-1, domain.geometry.dim))
w_inf_norm = np.linalg.norm(w_values, ord=np.inf)

# Define temporal parameters
CFL = 0.5
t = 0  # Start time
T = 1  # Final time
dt = CFL*hmax/(w_inf_norm)
num_steps = int(np.ceil(T/dt))
Cm = 0.5
eps = 1e-8
Cvel = 0.25
CRV = 1.0

#si = SI(1, domain, eps)
rv = RV(Cvel, CRV, domain)

# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
# xdmf.write_function(uh, t)

# Variational problem and solver
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(0))

a = u * v * ufl.dx + 0.5 * dt * ufl.dot(w, ufl.grad(u)) * v * ufl.dx
L = u_n * v * ufl.dx - 0.5 * dt * ufl.dot(w, ufl.grad(u_n)) * v * ufl.dx

# Preparing linear algebra structures for time dep. problems
bilinear_form = fem.form(a)
linear_form = fem.form(L)

# A does not change through time, but b does
A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
b = create_vector(linear_form)

# Can no longer use LinearProblem to solve since we already
# assembled a into matrix A. Therefore, create linear algebra solver with petsc4py
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

""" First, project hk in DG(0) on h_h in Lagrange(1) """
h_CG = get_nodal_h(domain, degree)

""" Take on GFEM step for residual calculation """
t += dt

# Update the right hand side reusing the initial vector
with b.localForm() as loc_b:
    loc_b.set(0)
assemble_vector(b, linear_form)

# Apply Dirichlet boundary condition to the vector
apply_lifting(b, [bilinear_form], [[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc])

# Solve linear problem
solver.solve(b, uh.x.petsc_vec)
uh.x.scatter_forward()

# Update solution at previous time step (u_n)
u_n.x.array[:] = uh.x.array

# Write solution to file

# vtx_eps = io.VTXWriter(domain.comm, f"{location_data}/discont_epsilon_degree_{degree}_{fraction}.bp", epsilon, engine="BP4")
# vtx_sol = io.VTXWriter(domain.comm, f"{location_data}/discont_sol_degree_{degree}_{fraction}.bp", u_n, engine="BP4")

#vtx_eps = io.VTXWriter(domain.comm, f"{location_data}/discont_epsilon_degree_{degree}_{fraction}.bp", epsilon, engine="BP4")
#tx_sol = io.VTXWriter(domain.comm, f"{location_data}/discont_sol_degree_{degree}_{fraction}.bp", u_n, engine="BP4")

xdmf_eps = io.XDMFFile(domain.comm, f"{location_data}/RV_epsilon_degree_{degree}_{fraction}.xdmf", "w", encoding=io.XDMFFile.Encoding.ASCII)
xdmf_eps.write_mesh(domain)
xdmf_sol = io.XDMFFile(domain.comm, f"{location_data}/RV_sol_degree_{degree}_{fraction}.xdmf", "w", encoding=io.XDMFFile.Encoding.ASCII)
xdmf_sol.write_mesh(domain)

V_vis = fem.functionspace(domain, ("Lagrange", 1))
uh_vis = fem.Function(V_vis)
epsilon_vis = fem.Function(V_vis)
uh_vis.interpolate(u_n)
epsilon_vis.interpolate(epsilon)
xdmf_sol.write_function(uh_vis, t)
xdmf_eps.write_function(epsilon_vis, t)


# Visualization of time dep. problem using pyvista
if PLOT:
    # pyvista.start_xvfb()

    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

    plotter = pyvista.Plotter(off_screen=True)
    plotter.open_gif(f"{location_figures}/RV_node.gif", fps=10)

    grid.point_data["uh"] = uh.x.array
    warped = grid.warp_by_scalar("uh", factor=1)

    viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
    sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                position_x=0.1, position_y=0.8, width=0.8, height=0.1)

    renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
                                cmap=viridis, scalar_bar_args=sargs,
                                clim=[0, max(uh.x.array)])
    
    # Configure 2D plotter for epsilon
    plotter_epsilon = pyvista.Plotter(off_screen=False)
    plotter_epsilon.open_gif("epsilon_2d.gif", fps=10)

    grid_epsilon = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
    grid_epsilon.point_data["epsilon"] = epsilon.x.array

    viridis_epsilon = mpl.colormaps.get_cmap("viridis").resampled(25)
    sargs_epsilon = dict(title="Epsilon", title_font_size=25, label_font_size=20, fmt="%.2e",
                        color="black", position_x=0.1, position_y=0.8, width=0.8, height=0.1)

    plotter_epsilon.add_mesh(grid_epsilon, show_edges=True, cmap=viridis_epsilon,
                            scalar_bar_args=sargs_epsilon,
                            clim=[np.min(epsilon.x.array), np.max(epsilon.x.array)])
    
    plotter_epsilon.view_xy()

# """ Then time loop """
for i in tqdm(range(num_steps-1)):
    t += dt

    a_R = u * v * ufl.dx
    L_R = 1/dt * u_n * v * ufl.dx - 1/dt * u_old * v * ufl.dx + ufl.dot(w, ufl.grad(u_n)) * v * ufl.dx

    # Solve linear system
    problem = LinearProblem(a_R, L_R, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    Rh = problem.solve() # returns dolfinx.fem.Function
    #Rh.x.array[:] = Rh.x.array / np.max(u_n.x.array - np.mean(u_n.x.array))

    #epsilon = fem.Function(V)
    #epsilon_placeholder = rv.get_epsilon_linear(uh, u_n, w, Rh, h_CG, node_patches, degree)
    epsilon = rv.get_epsilon_linear_simple(w, Rh, u_n, h_CG, degree)
    #epsilon.x.array[:] = epsilon_placeholder.x.array[:]

    a = u * v * ufl.dx + 0.5 * dt * ufl.dot(w, ufl.grad(u)) * v * ufl.dx + 0.5 * epsilon * dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx - 0.5 * dt * ufl.dot(w, ufl.grad(u_n)) * v * ufl.dx - 0.5 * epsilon * dt * ufl.dot(ufl.grad(u_n), ufl.grad(v)) * ufl.dx

    # Preparing linear algebra structures for time dep. problems
    bilinear_form = fem.form(a)
    linear_form = fem.form(L)

    # A does not change through time, but b does
    A = assemble_matrix(bilinear_form, bcs=[bc])
    A.assemble()
    b = create_vector(linear_form)

    solver.setOperators(A)
    """ Rest """
    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])
    
    # Solve linear problem
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Save previous solution
    u_old.x.array[:] = u_n.x.array
    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array

    # # Write solution to file
    # vtx_eps.write(t)
    # vtx_sol.write(t)


    # Write solution to file
    uh_vis.interpolate(uh)
    epsilon_vis.interpolate(epsilon)
    xdmf_eps.write_function(epsilon_vis,t)
    xdmf_sol.write_function(uh_vis,t)
    # Update plot
    if PLOT:
        new_warped = grid.warp_by_scalar("uh", factor=1)
        warped.points[:, :] = new_warped.points
        warped.point_data["uh"][:] = uh.x.array

        grid_epsilon.point_data["epsilon"][:] = epsilon.x.array
        plotter_epsilon.update_scalar_bar_range([np.min(epsilon.x.array), np.max(epsilon.x.array)])
        plotter_epsilon.write_frame()

        plotter.write_frame()

if PLOT:
    plotter.close()
xdmf_sol.close()
xdmf_eps.close()

# pde.plot_pv_2d(domain, mesh_size, uh, f'Solution at t = {T} with RV', 'lin_adv_rv', location_figures)
# pde.plot_pv_3d(domain, mesh_size, uh, f'Solution at t = {T} with RV', 'lin_adv_rv_3d', location_figures)