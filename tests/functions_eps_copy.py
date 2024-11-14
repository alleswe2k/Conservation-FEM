import matplotlib as mpl
import pyvista
import ufl
import numpy as np
from tqdm import tqdm

from petsc4py import PETSc
from mpi4py import MPI

import gmsh
from dolfinx.io import gmshio

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

PLOT = True

class PDE_solver:
    def __init__(self, initial_condition, f):
        self.init_cond = initial_condition
        self.velocity_field = f
        self.initialized = False

    def create_mesh(self, hmax):
        gdim = 2
        if self.initialized:
            gmsh.finalize()
        self.initialized = True
        gmsh.initialize()
        self.membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(gdim, [self.membrane], 1)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmax)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)
        gmsh.model.mesh.generate(gdim)

        gmsh_model_rank = 0
        mesh_comm = MPI.COMM_WORLD
        domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)
        return domain
    
    def create_vector(self, name, space, interpolation) -> fem.Function:
        vec = fem.Function(space)
        vec.name = name
        vec.interpolate(interpolation)
        return vec

    def boundary_condition(self, domain) -> fem.dirichletbc:
        fdim = domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
        bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(self.V, fdim, boundary_facets), self.V)
        return bc
    
    def create_solver(self, domain, A):
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
        return solver

    def solve(self, hmax, save=True):
        self.domain = self.create_mesh(hmax)

        self.V = fem.functionspace(self.domain, ("Lagrange", 1))
        self.W = fem.functionspace(self.domain, ("Lagrange", 1, (self.domain.geometry.dim, ))) # Lagrange 2 in documentation

        u_n = self.create_vector(name='u_n', space=self.V, interpolation=self.init_cond)
        w = self.create_vector(name='w', space=self.W, interpolation=self.velocity_field)
        uh = self.create_vector(name='uh', space=self.V, interpolation=self.init_cond)

        w_values = w.x.array.reshape((-1, self.domain.geometry.dim))
        w_inf_norm = np.linalg.norm(w_values, ord=np.inf)

        CFL = 0.5
        t = 0  # Start time
        T = 1  # Final time
        dt = CFL*hmax/w_inf_norm
        num_steps = int(np.ceil(T/dt))

        bc = self.boundary_condition(self.domain) 

        # Time-dependent output
        if save:
            xdmf = io.XDMFFile(self.domain.comm, "linear_advection.xdmf", "w")
            xdmf.write_mesh(self.domain)

        # Variational problem and solver
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        f = fem.Constant(self.domain, PETSc.ScalarType(0))
        a = u * v * ufl.dx + 0.5 * dt * ufl.dot(w, ufl.grad(u)) * v * ufl.dx
        L = u_n * v * ufl.dx - 0.5 * dt * ufl.dot(w, ufl.grad(u_n)) * v * ufl.dx

        # Preparing linear algebra structures for time dep. problems
        bilinear_form = fem.form(a)
        linear_form = fem.form(L)

        # A does not change through time, but b does
        A = assemble_matrix(bilinear_form, bcs=[bc])
        A.assemble()
        b = create_vector(linear_form)

        solver = self.create_solver(self.domain, A)

        # Updating the solution and rhs per time step
        for _ in tqdm(range(num_steps)):
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
            if save:
                xdmf.write_function(uh, t)

        if save:
            xdmf.close()
        self.uh = uh

    def calc_error(self):
        """ Compute L2 error and error at nodes """
        u_ex = self.create_vector(name='u_ex', space=self.V, interpolation=self.init_cond)
        error_L2 = np.sqrt(self.domain.comm.allreduce(fem.assemble_scalar(fem.form((self.uh - u_ex)**2 * ufl.dx)), op=MPI.SUM)) 
        return error_L2
    
    def calc_convergence(self):
        hmaxes = [1/4, 1/8, 1/16, 1/32]
        l2_errors = []
        for hmax in hmaxes:
            print(hmax)
            self.solve(hmax, save=False)
            l2_errors.append(self.calc_error())
        print(l2_errors)

        fitted_error = np.polyfit(np.log10(hmaxes), np.log10(l2_errors), 1)
        return fitted_error
    
    def plot(self):
        # TODO: Fixa denna
        pyvista.start_xvfb()

        # Extract topology from mesh and create pyvista mesh
        topology, cell_types, x = plot.vtk_mesh(self.V)
        grid = pyvista.UnstructuredGrid(topology, cell_types, x)

        # Set deflection values and add it to plotter
        grid.point_data["u"] = self.uh.x.array
        warped = grid.warp_by_scalar("u", factor=25)

        plotter = pyvista.Plotter()
        plotter.add_mesh(warped, show_edges=True, show_scalar_bar=True, scalars="u")
        plotter.show()
