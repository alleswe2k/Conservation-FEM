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
DISCONT = False
script_dir = os.path.dirname(os.path.abspath(__file__))
location_figures = os.path.join(script_dir, f"Figures/GFEM")
location_data = os.path.join(script_dir, f"Data/GFEM")

pde = PDE_plot()
# print(PETSc.ScalarType)

# def error_L2(uh, u_ex, degree_raise=3):
#     # Create higher order function space
#     degree = uh.function_space.ufl_element().degree
#     family = uh.function_space.ufl_element().family_name
#     mesh = uh.function_space.mesh
#     W = fem.functionspace(mesh, (family, degree + degree_raise))
#     # Interpolate approximate solution
#     u_W = fem.Function(W)
#     u_W.interpolate(uh)

#     # Interpolate exact solution, special handling if exact solution
#     # is a ufl expression or a python lambda function
#     u_ex_W = fem.Function(W)
#     if isinstance(u_ex, ufl.core.expr.Expr):
#         u_expr = fem.Expression(u_ex, W.element.interpolation_points())
#         u_ex_W.interpolate(u_expr)
#     else:
#         u_ex_W.interpolate(u_ex)

#     # Compute the error in the higher order function space
#     e_W = fem.Function(W)
#     e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

#     # Integrate the error
#     error = fem.form(ufl.inner(e_W, e_W) * ufl.dx)
#     error_local = fem.assemble_scalar(error)
#     error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
#     return np.sqrt(error_global)

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
        # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmax)
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

        # print("Infinity norm of the velocity field w:", w_inf_norm)

        # Create boundary condition
        fdim = domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
        bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

        # Time-dependent output
        xdmf = io.XDMFFile(domain.comm, f"{location_data}/sol_{"discont" if DISCONT else "cont"}_D{degree}_h{fraction}.xdmf", "w")
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
        # a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        # L = (u_n + dt * f) * v * ufl.dx

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

        # Visualization of time dep. problem using pyvista
        if PLOT:
            # pyvista.start_xvfb()

            grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

            plotter = pyvista.Plotter()
            plotter.open_gif("linear_advection.gif", fps=10)

            grid.point_data["uh"] = uh.x.array
            warped = grid.warp_by_scalar("uh", factor=1)

            viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
            sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                        position_x=0.1, position_y=0.8, width=0.8, height=0.1)

            renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
                                        cmap=viridis, scalar_bar_args=sargs,
                                        clim=[0, max(uh.x.array)])
        
        V_vis = fem.functionspace(domain, ("Lagrange", 1))
        uh_vis = fem.Function(V_vis)
        uh_vis.interpolate(u_n)
        xdmf.write_function(uh_vis, t)

        # Updating the solution and rhs per time step
        for i in range(num_steps):
            if t+dt > T:
                dt = T-t
            t += dt
            # print(t)

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
            uh_vis.interpolate(uh)
            xdmf.write_function(uh_vis, t)
            # Update plot
            if PLOT:
                new_warped = grid.warp_by_scalar("uh", factor=1)
                warped.points[:, :] = new_warped.points
                warped.point_data["uh"][:] = uh.x.array
                plotter.write_frame()

        print(t)
        # pde_realtime_plot.close()
        xdmf.close()

        error_L2 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
        if domain.comm.rank == 0:
            print(f"L2-error: {error_L2:.2e}")
        L2_errors.append(error_L2)


        # error = error_L2(uh, u_ex)
        # print(f"L2-error: {error:.2e}")
        # L2_errors.append(error)

        gmsh.finalize()

        # pde.plot_pv(domain, fraction, epsilon, 'Epsilon', 'cont_epsilon_2d_SI', location_figures, plot_2d=True)
        # pde.plot_pv(domain, fraction, uh, f'Solution at t = {T} with SI', 'cont_lin_adv_SI', location_figures)
        # pde.plot_pv(domain, fraction, uh, f'Solution at t = {T} with SI', 'cont_lin_adv_SI_3d', location_figures)

    print("D=", degree, "Errors=", L2_errors)
    pde.plot_convergence(L2_errors, fractions, f"P{degree} convergence", f"conv_{"discont" if DISCONT else "cont"}_D{degree}", location_figures)
    hs = [1/fraction for fraction in fractions]
    L2_errors = np.array(L2_errors)
    hs = np.array(hs)
    print(L2_errors, hs)
    rates = np.log(L2_errors[1:] / L2_errors[:-1]) / np.log(hs[1:] / hs[:-1])
    print(f"Rates: {rates}")