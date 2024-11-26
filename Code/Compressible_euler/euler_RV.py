from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import pyvista

from dolfinx.fem import Constant, Function, functionspace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx.io import VTXWriter
from dolfinx.mesh import create_unit_square
from dolfinx.plot import vtk_mesh
from basix.ufl import element
from ufl import TestFunction, TrialFunction, grad, div, dot, dx, inner, as_vector, outer, Identity, lhs, rhs

# Mesh and time stepping parameters
mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
t = 0
T = 10
num_steps = 500
dt = T / num_steps

# Function space for conserved variables (density, momentum, energy)
V = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 1, shape=(5,)))

# Define trial and test functions
U = TrialFunction(V)
V_test = TestFunction(V)

# Define initial conditions
U_n = Function(V, name="U_n")  # Previous time step
U_ = Function(V, name="U")  # Current time step

# Define parameters
gamma = 1.4
f = Constant(mesh, PETSc.ScalarType([0, 0, 0, 0, 0]))  # External forces

# Define flux tensor
def flux(U):
    rho, m1, m2, e = U[0], U[1], U[2], U[4]
    u = as_vector([m1 / rho, m2 / rho])  # Velocity vector
    T = (e / rho - 0.5 * dot(u, u)) / (gamma - 1)  # Temperature
    p = (gamma - 1) * rho * T  # Pressure
    
    F_rho = rho * u
    F_momentum = outer(u, rho * u) + p * Identity(len(u))
    F_energy = (e + p) * u
    return as_vector([F_rho[0], F_rho[1], F_momentum[0], F_momentum[1], F_energy])

# Weak form: ∂U/∂t + div(F(U)) = 0
residual = inner((U - U_n) / dt, V_test) * dx + inner(div(flux(U)), V_test) * dx - inner(f, V_test) * dx
a = form(lhs(residual))
L = form(rhs(residual))

# Assemble matrix and vectors
A = assemble_matrix(a)
A.assemble()
b = create_vector(L)

# Define solver
solver = PETSc.KSP().create(mesh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.BCGS)
pc = solver.getPC()
pc.setType(PETSc.PC.Type.HYPRE)

# Boundary and initial conditions
def initial_conditions(x):
    values = np.zeros((5, x.shape[1]), dtype=PETSc.ScalarType)
    values[0] = 1.0  # Density
    values[1] = 0.1  # Momentum x
    values[2] = 0.0  # Momentum y
    values[4] = 2.5  # Total energy
    return values

U_n.interpolate(initial_conditions)

# Output setup
from pathlib import Path
folder = Path("results_euler")
folder.mkdir(exist_ok=True, parents=True)
vtx = VTXWriter(mesh.comm, folder / "euler.bp", U_, engine="BP4")

# Time-stepping loop
for i in range(num_steps):
    t += dt

    # Assemble the right-hand side
    with b.localForm() as loc:
        loc.set(0)
    assemble_vector(b, L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    # Solve linear system
    solver.solve(b, U_.x.petsc_vec)
    U_.x.scatter_forward()

    # Update for the next time step
    U_n.x.array[:] = U_.x.array[:]

    # Write to file
    vtx.write(t)

vtx.close()
A.destroy()
b.destroy()
solver.destroy()
