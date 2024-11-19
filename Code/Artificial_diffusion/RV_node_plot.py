import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, LinearProblem

from PDE_solver import PDE_solver

hmax = 1/16
pde_solve = PDE_solver()
domain = pde_solve.create_mesh_unit_disk(hmax)

V = fem.functionspace(domain, ("Lagrange", 1))
W = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, ))) # Lagrange 2 in documentation
DG0 = fem.functionspace(domain, ("DG", 0))
DG1 = fem.functionspace(domain, ("DG", 1))

def initial_condition(x, r0=0.25, x0_1=0.3, x0_2=0):
    return (x[0] - x0_1)**2 + (x[1] - x0_2)**2 <= r0**2

def velocity_field(x):
    return np.array([-2*np.pi*x[1], 2*np.pi*x[0]])

u_n = pde_solve.create_vector(V, 'u_n', initial_condition)
u_old = pde_solve.create_vector(V, 'u_old', initial_condition)
u_ex = pde_solve.create_vector(V, 'u_ex', initial_condition)
w = pde_solve.create_vector(W, 'w', velocity_field)
uh = pde_solve.create_vector(V, 'uh', initial_condition)

CFL = 0.5
Cvel = 0.25
CRV = 1.0
T = 1.0
t = 0.0

dt, num_steps = pde_solve.get_time_steps(domain, w, CFL, T, hmax)
bc = pde_solve.boundary_condition(domain, V)

# Time-dependent output
xdmf = io.XDMFFile(domain.comm, "RV_node.xdmf", "w")
xdmf.write_mesh(domain)

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

solver = pde_solve.create_solver_linear(domain, A)

""" First, project hk in DG(0) on h_h in Lagrange(1) """
h_DG = fem.Function(DG0)  # Cell-based function for hk values

cell_to_vertex_map = domain.topology.connectivity(domain.topology.dim, 0)
vertex_coords = domain.geometry.x

num_cells = domain.topology.index_map(domain.topology.dim).size_local
hk_values = np.zeros(num_cells)

for cell in range(num_cells):
    # Get the vertices of the current cell
    cell_vertices = cell_to_vertex_map.links(cell)
    coords = vertex_coords[cell_vertices]  # Coordinates of the vertices
    
    edges = [np.linalg.norm(coords[i] - coords[j]) for i in range(len(coords)) for j in range(i + 1, len(coords))]
    hk_values[cell] = min(edges) 

h_DG.x.array[:] = hk_values

h_trial = ufl.TrialFunction(V)
a_h = h_trial * v * ufl.dx
L_h = h_DG * v * ufl.dx

# Solve linear system
problem = LinearProblem(a_h, L_h,petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
h_CG = problem.solve() # returns dolfinx.fem.Function

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
xdmf.write_function(uh, t)

# """ Then time loop """
for i in range(num_steps-1):
    t += dt

    a_R = u * v * ufl.dx
    L_R = 1/dt * u_n * v * ufl.dx - 1/dt * u_old * v * ufl.dx + ufl.dot(w, ufl.grad(u_n)) * v * ufl.dx

    # Solve linear system
    problem = LinearProblem(a_R, L_R, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    Rh = problem.solve() # returns dolfinx.fem.Function
    Rh.x.array[:] = Rh.x.array / np.max(u_n.x.array - np.mean(u_n.x.array))

    epsilon = fem.Function(V)

    for node in range(Rh.x.array.size):
        hi = h_CG.x.array[node]
        Ri = Rh.x.array[node]
        w_values = w.x.array.reshape((-1, domain.geometry.dim))
        fi = w_values[node]
        fi_norm = np.linalg.norm(fi)
        epsilon.x.array[node] = min(Cvel * hi * fi_norm, CRV * hi ** 2 * np.abs(Ri))

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

    # Write solution to file
    xdmf.write_function(uh, t)
xdmf.close()

pde_solve.plot_solution(domain, uh, 'Uh_plot', 'Uh')
pde_solve.plot_solution(domain, Rh, 'Rh_plot', 'Rh')