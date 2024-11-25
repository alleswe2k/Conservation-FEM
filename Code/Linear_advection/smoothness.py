""" TODO: residual and artificial viscosity plot """
import matplotlib as mpl
import pyvista
import ufl
import numpy as np
import os

from petsc4py import PETSc
from mpi4py import MPI

import gmsh
from dolfinx.io import gmshio

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, LinearProblem

from Utils.PDE_plot import PDE_plot
from Utils.PDE_realtime_plot import PDE_realtime_plot
from Utils.SI import SI
from Utils.helpers import get_nodal_h

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
location_figures = os.path.join(script_dir, 'Figures/SI')
location_data = os.path.join(script_dir, 'Data/SI/solution.xdmf')

pde = PDE_plot()
# print(PETSc.ScalarType)

# Enable or disable real-time plotting
PLOT = True
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

V = fem.functionspace(domain, ("Lagrange", 1))
# domain.geometry.dim = (2, )
W = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, ))) # Lagrange 2 in documentation
DG0 = fem.functionspace(domain, ("DG", 0))

# def initial_condition(x, r0=0.25, x0_1=0.3, x0_2=0):
#     return 1/2*(1-np.tanh(((x[0]-x0_1)**2+(x[1]-x0_2)**2)/r0**2 - 1))
def initial_condition(x, r0=0.25, x0_1=0.3, x0_2=0):
    return ((x[0] - x0_1)**2 + (x[1] - x0_2)**2 <= r0**2) * 14*np.pi + ((x[0] - x0_1)**2 + (x[1] - x0_2)**2 > r0**2) * np.pi / 4

def velocity_field(x):
    return np.array([-2*np.pi*x[1], 2*np.pi*x[0]])

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

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
Cm = 0.5

si = SI(Cm, domain)

# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(np.pi/4), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Time-dependent output
xdmf = io.XDMFFile(domain.comm, location_data, "w")
xdmf.write_mesh(domain)

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
h_CG = get_nodal_h(domain)

""" Creat patch dictionary """
# Dictionary to store adjacent nodes for each node
node_patches = si.get_patch_dictionary()

""" Assemble stiffness matrix, obtain element values """
a_stiffness = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
stiffness_matrix = assemble_matrix(fem.form(a_stiffness), bcs=[bc])
stiffness_matrix.assemble()

epsilon = fem.Function(V)

# Visualization of time dep. problem using pyvista
pde_realtime_plot = PDE_realtime_plot(location_figures, uh, epsilon, V)

# """ Then time loop """
for i in range(num_steps):
    t += dt

    # print(max(epsilon.x.array), min(epsilon.x.array))
    epsilon = si.get_epsilon_linear(w, node_patches, h_CG, u_n, stiffness_matrix)

     # Update plot
    pde_realtime_plot.update_plot(uh, epsilon)
    input()

    a = u * v * ufl.dx + 0.5 * dt * ufl.dot(w, ufl.grad(u)) * v * ufl.dx + 0.5 * epsilon * dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx - 0.5 * dt * ufl.dot(w, ufl.grad(u_n)) * v * ufl.dx - 0.5 * epsilon * dt * ufl.dot(ufl.grad(u_n), ufl.grad(v)) * ufl.dx

    # Preparing linear algebra structures for time dep. problems
    bilinear_form = fem.form(a)
    linear_form = fem.form(L)

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

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array

    # Write solution to file
    xdmf.write_function(uh, t)

pde_realtime_plot.close()
xdmf.close()

error_L2 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
if domain.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")

pde.plot_pv_2d(domain, fraction, epsilon, 'Epsilon', 'epsilon_2d', location=location_figures)