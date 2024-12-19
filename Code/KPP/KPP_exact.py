import matplotlib as mpl
import pyvista
import ufl
import numpy as np
import os 
import tqdm.autonotebook

from petsc4py import PETSc
from mpi4py import MPI

import gmsh
from dolfinx.io import gmshio

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import (VTXWriter, distribute_entity_data, gmshio)

from Utils.PDE_plot import PDE_plot
from Utils.RV import RV
from Utils.SI import SI
from Utils.helpers import get_nodal_h

script_dir = os.path.dirname(os.path.abspath(__file__))

location_fig = os.path.join(script_dir, 'Figures/RV') # location = './Figures'
location_data = os.path.join(script_dir, 'Data') # location = './Figures'

pde = PDE_plot()
gmsh.initialize()

membrane = gmsh.model.occ.addRectangle(-2,-2,0,4,4)
gmsh.model.occ.synchronize()

gdim = 2
gmsh.model.addPhysicalGroup(gdim, [membrane], 1)

hmax = 1/64
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmax)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)
gmsh.model.mesh.generate(gdim)

gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

pde.plot_grid(domain, location_fig)

V = fem.functionspace(domain, ("Lagrange", 1))
DG0 = fem.functionspace(domain, ("DG", 0))

def initial_condition(x):
    return (x[0]**2 + x[1]**2 <= 1) * 14*np.pi/4 + (x[0]**2 + x[1]**2 > 1) * np.pi/4

def velocity_field(u):
    # Apply nonlinear operators correctly to the scalar function u
    return ufl.as_vector([ufl.cos(u), -ufl.sin(u)])

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

u_old = fem.Function(V)
u_old.name = "u_old"
u_old.interpolate(initial_condition)

u_old_old = fem.Function(V)
u_old_old.name = "u_old_old"
u_old_old.interpolate(initial_condition)


CFL = 0.5
t = 0  # Start time
T = 1.0  # Final time
dt = 0.01
num_steps = int(np.ceil(T/dt))
Cvel = 0.5
CRV = 4.0

rv = RV(Cvel, CRV, domain)
si = SI(1, domain, 1e-8 )
node_patches = si.get_patch_dictionary()

# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(np.pi/4), fem.locate_dofs_topological(V, fdim, boundary_facets), V)
bc0 = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

RH = fem.Function(V)
RH.name = "RH"
RH.interpolate(lambda x: np.full(x.shape[1], 0, dtype = np.float64))

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

h_CG = get_nodal_h(domain)

# Time-dependent output
# with XDMFFile(MPI.COMM_WORLD, "KPP_exact.xdmf", "w") as xdmf:
#     xdmf_u.write_mesh(domain)
#     xdmf_u.write_function(uh)
xdmf_u = io.XDMFFile(domain.comm, location_data + '/KPP_exact.xdmf', "w", encoding=io.XDMFFile.Encoding.ASCII)
xdmf_u.write_mesh(domain)

# xdmf_e = io.XDMFFile(domain.comm, location_data + '/KPP_epsilon.xdmf', "w",  encoding=io.XDMFFile.Encoding.ASCII)
# xdmf_e.write_mesh(domain)

# vtx_u = VTXWriter(domain.comm, location_data + "/rv_exact.bp", [uh], engine="BP4")
# vtx_u.write(t)

progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=num_steps)
for i in range(num_steps):
    progress.update(1)
    t += dt

    """ BDF2 """
    F_R = (RH*v*ufl.dx - 
           3/(2*dt)*u_n*v *ufl.dx +
           4/(2*dt)*u_old*v*ufl.dx - 
           1/(2*dt)*u_old_old*v*ufl.dx - 
           ufl.dot(velocity_field(u_n), ufl.grad(u_n))*v*ufl.dx)
    R_problem = NonlinearProblem(F_R, RH, bcs=[bc0])
    # Rh = R_problem.solve()

    Rh_problem = NewtonSolver(MPI.COMM_WORLD, R_problem)
    Rh_problem.convergence_criterion = "incremental"
    Rh_problem.max_it = 100  # Increase maximum number of iterations
    Rh_problem.rtol =  1e-1
    Rh_problem.report = True

    n, converged = Rh_problem.solve(RH)

    epsilon = rv.get_epsilon_nonlinear(uh, u_n, velocity_field, RH, h_CG, node_patches)

    F = (uh*v *ufl.dx - u_n*v *ufl.dx + 
        0.5*dt*ufl.dot(velocity_field(uh), ufl.grad(uh))*v*ufl.dx + 
        0.5*dt*ufl.dot(velocity_field(u_n), ufl.grad(u_n))*v*ufl.dx + 
        0.5*dt*epsilon*ufl.dot(ufl.grad(uh), ufl.grad(v))*ufl.dx +
        0.5*dt*epsilon*ufl.dot(ufl.grad(u_n), ufl.grad(v))*ufl.dx)
    
    problem = NonlinearProblem(F, uh, bcs = [bc])
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.max_it = 100  # Increase maximum number of iterations
    solver.rtol =  1e-4
    solver.report = True

    # Solve linear problem
    n, converged = solver.solve(uh)
    assert (converged)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_old_old.x.array[:] = u_old.x.array
    u_old.x.array[:] = u_n.x.array
    u_n.x.array[:] = uh.x.array

    # vtx_u.write(t)

    xdmf_u.write_function(uh, t)
    # xdmf_e.write_function(epsilon, t)
progress.close()
# vtx_u.close()
xdmf_u.close()
# xdmf_e.close()

    


