""" The working version! """
import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

import gmsh
from dolfinx.io import gmshio

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, LinearProblem

# Enable or disable real-time plotting
PLOT = True
# Creating mesh
gmsh.initialize()

membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
gmsh.model.occ.synchronize()

gdim = 2
gmsh.model.addPhysicalGroup(gdim, [membrane], 1)

hmax = 1/16 # 0.05 in example
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmax)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)
gmsh.model.mesh.generate(gdim)

gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

V = fem.functionspace(domain, ("Lagrange", 1))
# domain.geometry.dim = (2, )
W = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, ))) # Lagrange 2 in documentation

DG = fem.functionspace(domain, ("DG", 1)) # Lagrange 2 in documentation


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

w_values = w.x.array.reshape((-1, domain.geometry.dim))
w_inf_norm = np.linalg.norm(w_values, ord=np.inf)
# TODO: This is probably more correct
# w_norms = np.linalg.norm(w_values, axis=1)
# w_inf_norm = np.max(w_norms)

# Define temporal parameters
CFL = 0.5
t = 0  # Start time
T = 1.0  # Final time
dt = CFL*hmax/w_inf_norm
num_steps = int(np.ceil(T/dt))
Cvel = 0.25
CRV = 1.0

# print("Infinity norm of the velocity field w:", w_inf_norm)

# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Time-dependent output
xdmf = io.XDMFFile(domain.comm, "RV.xdmf", "w")
xdmf.write_mesh(domain)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
# xdmf.write_function(uh, t)

h = ufl.CellDiameter(domain)

h_DG = fem.Function(DG)
num_cells = domain.topology.index_map(domain.topology.dim).size_local


for cell in range(num_cells):
    # TODO: DG instead of V?
    loc2glb = DG.dofmap.cell_dofs(cell)
    x = DG.tabulate_dof_coordinates()[loc2glb]
    print(x)
    edges = [np.linalg.norm(x[i] - x[j]) for i in range(3) for j in range(i+1, 3)]
    hk = min(edges) # NOTE: Max gives better convergence
    h_DG.x.array[loc2glb] = hk

# # Variational problem and solver
# u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
# f = fem.Constant(domain, PETSc.ScalarType(0))

# # Define time-discretized residual R
# R = (u - u_n) / dt + ufl.div(w * u) - ufl.div(ufl.grad(u))
# # Calculate L2 norm of residual for each cell
# R_norm = ufl.sqrt(ufl.dot(R, R))
# print("R norm:", R_norm)
# alpha = 0.01  # Tuning parameter for viscosity
# epsilon = alpha * h * R_norm
# print("Epsilon:", epsilon)

# a = u * v * ufl.dx + 0.5 * dt * ufl.dot(w, ufl.grad(u)) * v * ufl.dx
# L = u_n * v * ufl.dx - 0.5 * dt * ufl.dot(w, ufl.grad(u_n)) * v * ufl.dx

# # Preparing linear algebra structures for time dep. problems
# bilinear_form = fem.form(a)
# linear_form = fem.form(L)

# # A does not change through time, but b does
# A = assemble_matrix(bilinear_form, bcs=[bc])
# A.assemble()
# b = create_vector(linear_form)

# # Can no longer use LinearProblem to solve since we already
# # assembled a into matrix A. Therefore, create linear algebra solver with petsc4py
# solver = PETSc.KSP().create(domain.comm)
# solver.setOperators(A)
# solver.setType(PETSc.KSP.Type.PREONLY)
# solver.getPC().setType(PETSc.PC.Type.LU)

# # Visualization of time dep. problem using pyvista
# if PLOT:
#     # pyvista.start_xvfb()

#     grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

#     plotter = pyvista.Plotter()
#     plotter.open_gif("RV.gif", fps=10)

#     grid.point_data["uh"] = uh.x.array
#     warped = grid.warp_by_scalar("uh", factor=1)

#     viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
#     sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
#                 position_x=0.1, position_y=0.8, width=0.8, height=0.1)

#     renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
#                                 cmap=viridis, scalar_bar_args=sargs,
#                                 clim=[0, max(uh.x.array)])

# """ Take on GFEM step for residual calculation """
# t += dt

# # Update the right hand side reusing the initial vector
# with b.localForm() as loc_b:
#     loc_b.set(0)
# assemble_vector(b, linear_form)

# # Apply Dirichlet boundary condition to the vector
# apply_lifting(b, [bilinear_form], [[bc]])
# b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
# set_bc(b, [bc])

# # Solve linear problem
# solver.solve(b, uh.x.petsc_vec)
# uh.x.scatter_forward()

# # Update solution at previous time step (u_n)
# u_n.x.array[:] = uh.x.array

# # Write solution to file
# xdmf.write_function(uh, t)

# """ Then time loop """
# for i in range(num_steps):
#     t += dt

#     a_R = u * v * ufl.dx
#     L_R = 1/dt * u_n * v * ufl.dx - 1/dt * u_old * v * ufl.dx + ufl.dot(w, ufl.grad(u_n)) * v * ufl.dx

#     # Solve linear system
#     problem = LinearProblem(a_R, L_R, petsc_options={"kst_type": "preonly", "pc_type": "lu"})
#     Rh = problem.solve() # returns dolfinx.fem.Function
#     Rh.x.array[:] = Rh.x.array / np.max(u_n.x.array - np.mean(u_n.x.array))
#     # print(np.max(u_n.x.array - np.mean(u_n.x.array)))
#     # print(u_n.x.array)

#     epsilon = fem.Function(V) # TODO: Test DG space
#     num_cells = domain.topology.index_map(domain.topology.dim).size_local

#     for cell in range(num_cells):
#         loc2glb = V.dofmap.cell_dofs(cell)
#         Rk = np.max(np.abs(Rh.x.array[loc2glb]))
#         w_values = w.x.array.reshape((-1, domain.geometry.dim))
#         Bk = np.max(np.sqrt(np.sum(w_values[loc2glb]**2, axis=1)))
        
#         x = V.tabulate_dof_coordinates()[loc2glb]
#         edges = [np.linalg.norm(x[i] - x[j]) for i in range(3) for j in range(i+1, 3)]
#         hk = max(edges)

#         epsilon_k = min(Cvel * hk * Bk, CRV * hk **2 * Rk)
#         for dof in loc2glb:
#             # TODO: Try = instead of +=
#             epsilon.x.array[dof] = epsilon_k

#     a = u * v * ufl.dx + 0.5 * dt * ufl.dot(w, ufl.grad(u)) * v * ufl.dx + 0.5 * epsilon * dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
#     L = u_n * v * ufl.dx - 0.5 * dt * ufl.dot(w, ufl.grad(u_n)) * v * ufl.dx - 0.5 * epsilon * dt * ufl.dot(ufl.grad(u_n), ufl.grad(v)) * ufl.dx

#     # Preparing linear algebra structures for time dep. problems
#     bilinear_form = fem.form(a)
#     linear_form = fem.form(L)

#     # A does not change through time, but b does
#     A = assemble_matrix(bilinear_form, bcs=[bc])
#     A.assemble()
#     b = create_vector(linear_form)

#     solver = PETSc.KSP().create(domain.comm)
#     solver.setOperators(A)
#     solver.setType(PETSc.KSP.Type.PREONLY)
#     solver.getPC().setType(PETSc.PC.Type.LU)
#     """ Rest """
#     # Update the right hand side reusing the initial vector
#     with b.localForm() as loc_b:
#         loc_b.set(0)
#     assemble_vector(b, linear_form)

#     # Apply Dirichlet boundary condition to the vector
#     apply_lifting(b, [bilinear_form], [[bc]])
#     b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
#     set_bc(b, [bc])
    
#     # Solve linear problem
#     solver.solve(b, uh.x.petsc_vec)
#     uh.x.scatter_forward()

#     # Save previous solution
#     u_old.x.array[:] = u_n.x.array
#     # Update solution at previous time step (u_n)
#     u_n.x.array[:] = uh.x.array

#     # Write solution to file
#     xdmf.write_function(uh, t)
#     # Update plot
#     if PLOT:
#         new_warped = grid.warp_by_scalar("uh", factor=1)
#         warped.points[:, :] = new_warped.points
#         warped.point_data["uh"][:] = uh.x.array
#         plotter.write_frame()

# if PLOT:
#     plotter.close()
# xdmf.close()

# error_L2 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
# if domain.comm.rank == 0:
#     print(f"L2-error: {error_L2:.2e}")