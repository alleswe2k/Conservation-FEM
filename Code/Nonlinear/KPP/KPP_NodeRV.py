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


PLOT = True
gmsh.initialize()

membrane = gmsh.model.occ.addRectangle(-2,-2,0,4,4)
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
DG0 = fem.functionspace(domain, ("DG", 0))
DG1 = fem.functionspace(domain, ("DG", 1))

def initial_condition(x):
    # return np.where(x[0]**2 + x[1]**2 <= 1,  14 *np.pi / 4, np.pi / 4)
    return (x[0]**2 + x[1]**2 <= 1) * 14*np.pi/4 + (x[0]**2 + x[1]**2 > 1) * 0
def velocity_field(u):
    # Apply nonlinear operators correctly to the scalar function u
    return ufl.as_vector([ufl.cos(u), -ufl.sin(u)])

def Res_condition(x):
    return PETSc.ScalarType(0)


u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

u_old = fem.Function(V)
u_old.name = "u_old"
u_old.interpolate(initial_condition)



CFL = 0.5
t = 0  # Start time
T = 1.0  # Final time
dt = 0.01
num_steps = int(np.ceil(T/dt))
Cvel = 0.25
CRV = 1.0


# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Time-dependent output
xdmf = io.XDMFFile(domain.comm, "Code/Nonlinear/KPP/Output/testing.xdmf", "w")
xdmf.write_mesh(domain)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

RH = fem.Function(V)
RH.name = "uh"
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
xdmf.write_function(uh, t)


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
    xdmf.write_function(uh, t)
    # Update plot
    if PLOT:
        new_warped = grid.warp_by_scalar("uh", factor=1)
        warped.points[:, :] = new_warped.points
        warped.point_data["uh"][:] = uh.x.array
        plotter.write_frame()
if PLOT:
    plotter.close()
xdmf.close()

