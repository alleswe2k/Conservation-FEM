import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

import gmsh
from dolfinx.io import gmshio

from dolfinx import fem, mesh, io, plot, nls, log
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from PDE_solver import PDE_solver


pde = PDE_solver()
PLOT = False
# gmsh.initialize()
# mesh_comm = MPI.COMM_WORLD

# membrane = gmsh.model.occ.addRectangle(-1,-1, 0, 1,1)
# # membrane = mesh.create_unit_square(MPI.COMM_WORLD, 1000, 1000)
# gmsh.model.occ.synchronize()

# gdim = 2
# gmsh.model.addPhysicalGroup(gdim, [membrane], 1)

# hmax = 1/16 # 0.05 in example
# gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmax)
# gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)
# gmsh.model.mesh.generate(gdim)

# gmsh_model_rank = 0
# domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [100, 100], cell_type=mesh.CellType.triangle)

V = fem.functionspace(domain, ("Lagrange", 1))
DG0 = fem.functionspace(domain, ("DG", 0))
DG1 = fem.functionspace(domain, ("DG", 1))


def velocity_field(u):
    # Apply nonlinear operators correctly to the scalar function u
    return ufl.as_vector([u,u])

def exact_solution(x): 
    t = 1

    u = np.zeros_like(x[0])  # Initialize the solution array with zeros
    
    # First condition
    mask1 = x[0] <= (1/2 - 3 * t / 5)
    u = np.where(mask1 & (x[1] > (1/2 + 3 * t / 20)), -0.2, u)
    u = np.where(mask1 & (x[1] <= (1/2 + 3 * t / 20)), 0.5, u)

    # Second condition
    mask2 = (1/2 - t / 4 <= x[0]) & (x[0] <= (1/2 + t / 2))
    u = np.where(mask2 & (x[1] > (-8 * x[0] / 7 + 15 / 14 - 15 * t / 28)), -1, u)
    u = np.where(mask2 & (x[1] <= (-8 * x[0] / 7 + 15 / 14 - 15 * t / 28)), 0.5, u)

    # Third condition
    mask3 = (1/2 - t / 4 <= x[0]) & (x[0] <= (1/2 + t / 2))
    u = np.where(mask3 & (x[1] > (x[0] / 6 + 5 / 12 - 5 * t / 24)), -1, u)
    u = np.where(mask3 & (x[1] <= (x[0] / 6 + 5 / 12 - 5 * t / 24)), 0.5, u)

    # Fourth condition
    mask4 = (1/2 + t / 2 <= x[0]) & (x[0] <= (1/2 + 4 * t / 5))
    u = np.where(mask4 & (x[1] > (x[0] - 5 / (18 * t) * (x[0] + t - 1/2)**2)), -1, u)
    u = np.where(mask4 & (x[1] <= (x[0] - 5 / (18 * t) * (x[0] + t - 1/2)**2)), (2 * x[0] - 1) / (2 * t), u)

    # Fifth condition
    mask5 = x[0] >= (1/2 + 4 * t / 5)
    u = np.where(mask5 & (x[1] > (1/2 - t / 10)), -1, u)
    u = np.where(mask5 & (x[1] <= (1/2 - t / 10)), 0.8, u)

    return u



def initial_condition(x):
    x0, x1 = x[0], x[1]  # Extract x0 and x1 from the input array

    # Initialize the solution array with zeros
    u = np.zeros_like(x0)
    # Apply the conditions
    u = np.where((x0 <= 0.5) & (x1 >= 0.5), -0.2, u)
    u = np.where((x0 > 0.5) & (x1 >= 0.5), -1.0, u)
    u = np.where((x0 <= 0.5) & (x1 < 0.5), 0.5, u)
    u = np.where((x0 > 0.5) & (x1 < 0.5), 0.8, u)
    return u


u_exact = fem.Function(V)
u_exact.name = "U Exact"
u_exact.interpolate(exact_solution)

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

u_old = fem.Function(V)
u_old.name = "u_old"
u_old.interpolate(initial_condition)



CFL = 0.5
t = 0  # Start time
T = 1  # Final time
dt = 0.01
num_steps = int(np.ceil(T/dt))
Cvel = 0.25
CRV = 1.0


# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(np.pi/4), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# # Time-dependent output
# xdmf = io.XDMFFile(domain.comm, "Code/Nonlinear/KPP/Output/testing.xdmf", "w")
# xdmf.write_mesh(domain)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

RH = fem.Function(V)
RH.name = "RH"
RH.interpolate(lambda x: np.full(x.shape[1], 0, dtype = np.float64))

# Variational problem and solver
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
F = (uh*v *ufl.dx -
     u_n*v *ufl.dx + 
     0.5*dt*ufl.dot(velocity_field(uh), ufl.grad(uh))*v*ufl.dx + 
     0.5*dt*ufl.dot(velocity_field(u_n), ufl.grad(u_n))*v*ufl.dx)


nonlin_problem = NonlinearProblem(F, uh, bcs = [bc])
nonlin_solver = NewtonSolver(MPI.COMM_WORLD, nonlin_problem)
nonlin_solver.max_it = 100  # Increase maximum number of iterations
nonlin_solver.rtol = 1e-1
nonlin_solver.report = True
if PLOT:
    # pyvista.start_xvfb()

    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

    plotter = pyvista.Plotter()
    plotter.open_gif("Code/Nonlinear/KPP/Output/KPP.gif", fps=10)

    grid.point_data["uh"] = uh.x.array
    warped = grid.warp_by_scalar("uh", factor=1)

    viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
    sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                position_x=0.1, position_y=0.8, width=0.8, height=0.1)

    renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
                                cmap=viridis, scalar_bar_args=sargs,
                                clim=[0, max(uh.x.array)])
    

h_DG = fem.Function(DG1)
num_cells = domain.topology.index_map(domain.topology.dim).size_local

for cell in range(num_cells):
    # TODO: DG instead of V?
    loc2glb = DG1.dofmap.cell_dofs(cell)
    x = DG1.tabulate_dof_coordinates()[loc2glb]
    edges = [np.linalg.norm(x[i] - x[j]) for i in range(3) for j in range(i+1, 3)]
    hk = min(edges) # NOTE: Max gives better convergence
    h_DG.x.array[loc2glb] = hk



v = ufl.TestFunction(V)
h_trial = ufl.TrialFunction(V)
a_h = h_trial * v * ufl.dx
L_h = h_DG * v * ufl.dx

# Solve linear system
lin_problem = LinearProblem(a_h, L_h, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
h_CG = lin_problem.solve() # returns dolfinx.fem.Function


#Take GFEM STEP
t += dt
n, converged = nonlin_solver.solve(uh)
assert (converged)
# uh.x.scatter_forward()

# Update solution at previous time step (u_n)
u_n.x.array[:] = uh.x.array
# Write solution to file
# xdmf.write_function(uh, t)



for i in range(num_steps -1):
    t += dt

    
    # a_R = u*v*ufl.dx
    # L_R = 1/dt*u_n * v * ufl.dx - 1/dt* u_old * v *ufl.dx + ufl.dot(velocity_field(u_n),ufl.grad(u_n))* v * ufl.dx
    # F_R = (a_R - L_R)


    F_R = (RH*v*ufl.dx - 1/dt*u_n*v *ufl.dx - 1/dt*u_old*v*ufl.dx +
        0.5*ufl.dot(velocity_field(u_n), ufl.grad(u_n))*v*ufl.dx + 
        0.5*ufl.dot(velocity_field(u_old), ufl.grad(u_old))*v*ufl.dx)
    R_problem = NonlinearProblem(F_R, RH, bcs = [bc])
    # Rh = R_problem.solve()

    Rh_problem = NewtonSolver(MPI.COMM_WORLD, R_problem)
    Rh_problem.convergence_criterion = "incremental"
    Rh_problem.max_it = 100  # Increase maximum number of iterations
    Rh_problem.rtol =  1e-1
    Rh_problem.report = True

    n, converged = Rh_problem.solve(RH)
    # n, converged = Rh.solve(uh)
    # assert (converged)
    RH.x.array[:] = RH.x.array / np.max(u_n.x.array - np.mean(u_n.x.array))
    epsilon = fem.Function(V)

    for node in range(RH.x.array.size):
        hi = h_CG.x.array[node]
        Ri = RH.x.array[node]
        w = uh.x.array[node]
        w = velocity_field(uh.x.array[node])
        fi = np.array(w, dtype = 'float')
        fi_norm = np.linalg.norm(fi)
        epsilon.x.array[node] = min(Cvel * hi * fi_norm, CRV * hi ** 2 * np.abs(Ri))
    
    F = (uh*v *ufl.dx - u_n*v *ufl.dx + 
        0.5*dt*ufl.dot(velocity_field(uh), ufl.grad(uh))*v*ufl.dx + 
        0.5*dt*ufl.dot(velocity_field(u_n), ufl.grad(u_n))*v*ufl.dx + 
        0.5*dt*epsilon*ufl.dot(ufl.grad(uh), ufl.grad(v))*ufl.dx +
        0.5*dt*epsilon*ufl.dot(ufl.grad(u_n), ufl.grad(v))*ufl.dx)
    
    problem = NonlinearProblem(F, uh, bcs = [bc])
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.max_it = 100  # Increase maximum number of iterations
    solver.rtol =  1e-1
    solver.report = True

    # Solve linear problem
    n, converged = solver.solve(uh)
    assert (converged)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_old.x.array[:] = u_n.x.array
    u_n.x.array[:] = uh.x.array

    # Write solution to file
    # xdmf.write_function(uh, t)
    # Update plot
    if PLOT:
        new_warped = grid.warp_by_scalar("uh", factor=1)
        warped.points[:, :] = new_warped.points
        warped.point_data["uh"][:] = uh.x.array
        plotter.write_frame()

pde.plot_solution(domain, u_exact, "exact_solution", "Exact Solution")

u_exact.interpolate(initial_condition)
pde.plot_solution(domain, u_exact, "initial_exact", "Initial Exact")


print(f'Error: {np.abs(u_exact.x.array - uh.x.array)}')

if PLOT:
    plotter.close()
# xdmf.close()


  


