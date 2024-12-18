import matplotlib as mpl
import pyvista
import ufl
import numpy as np
import os
from tqdm import tqdm

from petsc4py import PETSc
from mpi4py import MPI

import gmsh
from dolfinx.io import gmshio

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, LinearProblem

from Utils.PDE_plot import PDE_plot
from Utils.PDE_realtime_plot import PDE_realtime_plot
from Utils.SI import SI
from Utils.RV import RV
from Utils.helpers import get_nodal_h

import os
# Enable or disable real-time plotting
PLOT = False
DISCONT = True
STABILIZATION = "RV"
script_dir = os.path.dirname(os.path.abspath(__file__))
location_figures = os.path.join(script_dir, f"Figures/{STABILIZATION}")
location_data = os.path.join(script_dir, f"Data/{STABILIZATION}")

pde = PDE_plot()
# print(PETSc.ScalarType)

degrees = [1, 2, 3]
for degree in degrees:
    fractions = [4, 8, 16, 32]
    L2_errors = []
    for fraction in fractions:
        # Creating mesh
        gmsh.initialize()

        membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()

        gdim = 2
        gmsh.model.addPhysicalGroup(gdim, [membrane], 1)

        hmax = 1/fraction # 0.05 in example
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmax)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)
        gmsh.model.mesh.generate(gdim)

        gmsh_model_rank = 0
        mesh_comm = MPI.COMM_WORLD
        domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

        # degree = 3
        V = fem.functionspace(domain, ("Lagrange", degree))
        V_ex = fem.functionspace(domain, ("Lagrange", degree*2))
        # domain.geometry.dim = (2, )
        W = fem.functionspace(domain, ("Lagrange", degree, (domain.geometry.dim, ))) # Lagrange 2 in documentation
        DG0 = fem.functionspace(domain, ("DG", 0))

        if DISCONT:
            """ Discont. IC """
            def initial_condition(x, r0=0.25, x0_1=0.3, x0_2=0):
                return (x[0] - x0_1)**2 + (x[1] - x0_2)**2 <= r0**2
        else:
            """ Cont. IC """
            def initial_condition(x, r0=0.25, x0_1=0.3, x0_2=0):
                return 1/2*(1-np.tanh(((x[0]-x0_1)**2+(x[1]-x0_2)**2)/r0**2 - 1))

        def velocity_field(x):
            return np.array([-2*np.pi*x[1], 2*np.pi*x[0]])

        u_n = fem.Function(V)
        u_n.name = "u_n"
        u_n.interpolate(initial_condition)

        u_old = fem.Function(V)
        u_old.name = "u_old"
        u_old.interpolate(initial_condition)

        u_ex = fem.Function(V_ex)
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
        eps = 1e-8
        Cvel = 0.25
        CRV = 1.0

        si = SI(Cm, domain, eps)
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

        """ Creat patch dictionary """
        # Dictionary to store adjacent nodes for each node
        node_patches = si.get_patch_dictionary()

        """ Assemble stiffness matrix, obtain element values """
        a_stiffness = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        stiffness_matrix = assemble_matrix(fem.form(a_stiffness), bcs=[bc])
        stiffness_matrix.assemble()

        epsilon = fem.Function(V)
        numerator_func = fem.Function(V)

        # Time-dependent output
        # vtx_sol = io.VTXWriter(domain.comm, 
        #                         f"{location_data}/sol_{"discont" if DISCONT else "cont"}_D{degree}_h{fraction}.bp", 
        #                         u_n, engine="BP4")
        # vtx_eps = io.VTXWriter(domain.comm, 
        #                         f"{location_data}/eps_{"discont" if DISCONT else "cont"}_h{fraction}.bp", 
        #                         epsilon, engine="BP4")
        # xdmf_sol.write(t)
        # xdmf_eps.write(t)
        xdmf_sol = io.XDMFFile(domain.comm, f"{location_data}/sol_{"discont" if DISCONT else "cont"}_D{degree}_h{fraction}.xdmf", "w")
        xdmf_eps = io.XDMFFile(domain.comm, f"{location_data}/eps_{"discont" if DISCONT else "cont"}_h{fraction}.xdmf", "w")
        xdmf_sol.write_mesh(domain)
        xdmf_eps.write_mesh(domain)

        V_vis = fem.functionspace(domain, ("Lagrange", 1))
        uh_vis = fem.Function(V_vis)
        epsilon_vis = fem.Function(V_vis)
        uh_vis.interpolate(u_n)
        epsilon_vis.interpolate(epsilon)
        xdmf_sol.write_function(uh_vis, t)
        xdmf_eps.write_function(epsilon_vis, t)

        # Visualization of time dep. problem using pyvista
        # pde_realtime_plot = PDE_realtime_plot(location_figures, uh, epsilon, V, numerator_func)

        # """ Then time loop """
        for i in tqdm(range(num_steps)):
            t += dt

            if STABILIZATION == "RV":
                a_R = u * v * ufl.dx
                L_R = 1/dt * u_n * v * ufl.dx - 1/dt * u_old * v * ufl.dx + ufl.dot(w, ufl.grad(u_n)) * v * ufl.dx

                # Solve linear system
                problem = LinearProblem(a_R, L_R, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
                Rh = problem.solve() # returns dolfinx.fem.Function
                #Rh.x.array[:] = Rh.x.array / np.max(u_n.x.array - np.mean(u_n.x.array))

                #epsilon = fem.Function(V)
                epsilon = rv.get_epsilon_linear(uh, u_n, w, Rh, h_CG, node_patches, degree)
            elif STABILIZATION == "SI":
                # print(max(epsilon.x.array), min(epsilon.x.array))
                epsilon = si.get_epsilon_linear(w, node_patches, h_CG, u_n, stiffness_matrix, numerator_func, degree)

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
            u_old.x.array[:] = u_n.x.array
            u_n.x.array[:] = uh.x.array

            # Write solution to file
            uh_vis.interpolate(uh)
            epsilon_vis.interpolate(epsilon)
            xdmf_sol.write_function(uh_vis, t)
            xdmf_eps.write_function(epsilon_vis, t)
            # xdmf_sol.write(t)
            # xdmf_eps.write(t)

        # pde_realtime_plot.close()
        xdmf_sol.close()
        xdmf_eps.close()

        error_L2 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
        if domain.comm.rank == 0:
            print(f"L2-error: {error_L2:.2e}")
        
        L2_errors.append(error_L2)
        
        gmsh.finalize()

        # pde.plot_pv(domain, fraction, epsilon, 'Epsilon', 'cont_epsilon_2d_SI', location_figures, plot_2d=True)
        # pde.plot_pv(domain, fraction, uh, f'Solution at t = {T} with SI', 'cont_lin_adv_SI', location_figures)
        # pde.plot_pv(domain, fraction, uh, f'Solution at t = {T} with SI', 'cont_lin_adv_SI_3d', location_figures)

    pde.plot_convergence(L2_errors, fractions, f"P{degree} convergence", f"conv_{"discont" if DISCONT else "cont"}_D{degree}", location_figures)