""" The working version! """
""" TODO: Try assembling static part and adding dynamic part in loop """
import matplotlib as mpl
import pyvista
import ufl
import numpy as np
import matplotlib.pyplot as plt

from petsc4py import PETSc
from mpi4py import MPI

import gmsh
from dolfinx.io import gmshio

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, LinearProblem

from Utils.PDE_plot import PDE_plot
from Utils.SI import SI
from Utils.RV import RV

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
location_figures = os.path.join(script_dir, 'Figures/SI')

L2_errors = []
hmaxes = [1/4, 1/8, 1/16, 1/32]
for hmax in hmaxes:
    # Creating mesh
    gmsh.initialize()

    membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
    gmsh.model.occ.synchronize()

    gdim = 2
    gmsh.model.addPhysicalGroup(gdim, [membrane], 1)

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

    si = SI(1, domain)
    rv = RV(Cvel, CRV, domain)
    node_patches = si.get_patch_dictionary()

    # Create boundary condition
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

    # Define solution variable, and interpolate initial solution for visualization in Paraview
    uh = fem.Function(V)
    uh.name = "uh"
    uh.interpolate(initial_condition)

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

    v = ufl.TestFunction(V)

    h_trial = ufl.TrialFunction(V)
    a_h = h_trial * v * ufl.dx
    L_h = h_DG * v * ufl.dx

    # Solve linear system
    problem = LinearProblem(a_h, L_h, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
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

    """ Then time loop """
    for i in range(num_steps-1):
        t += dt

        a_R = u * v * ufl.dx
        L_R = 1/dt * u_n * v * ufl.dx - 1/dt * u_old * v * ufl.dx + ufl.dot(w, ufl.grad(u_n)) * v * ufl.dx

        # Solve linear system
        problem = LinearProblem(a_R, L_R, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        Rh = problem.solve() # returns dolfinx.fem.Function
        #Rh.x.array[:] = Rh.x.array / np.max(u_n.x.array - np.mean(u_n.x.array)) #----- 0.41870130343981926
        Rh.x.array[:] = rv.normalize_Rh_robust_ni(uh, Rh, node_patches) #------- 0.5813225474370259
        #Rh.x.array[:] = rv.normalize_Rh_robust_simple(uh, Rh, node_patches) #------- 0.5344210238866037
        #Rh.x.array[:] = rv.normalize_Rh(uh, Rh, node_patches) #-------0.5774394314048876
        #Rh.x.array[:] = Rh.x.array / np.linalg.norm(u_n.x.array - np.mean(u_n.x.array), ord=np.inf) #-------0.41870130343981926


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

    # Compute L2 error and error at nodes
    error_L2 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
    if domain.comm.rank == 0:
        print(f"L2-error: {error_L2:.2e}")
    
    L2_errors.append(float(error_L2))

    gmsh.finalize()

print(f'L2-errors:{L2_errors}')
fitted_error = np.polyfit(np.log10(hmaxes), np.log10(L2_errors), 1)
print(f'convergence: {fitted_error[0]}')

pde = PDE_plot()
pde.plot_convergence(L2_errors, 'RV-Nodal', 'rv_conv', location_figures)




