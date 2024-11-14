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
    def __init__(self, hmax, T, initial_condition):
        self.hmax = hmax
        self.T = T
        self.init_cond = initial_condition
        self.domain = self.create_mesh()

        self.V = fem.functionspace(self.domain, ("Lagrange", 1))
        self.W = fem.functionspace(self.domain, ("Lagrange", 1, (self.domain.geometry.dim, )))

    def create_mesh(self) -> gmsh:
        gmsh.initialize()
        membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        gdim = 2
        gmsh.model.addPhysicalGroup(gdim, [membrane], 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.hmax)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.hmax)
        gmsh.model.mesh.generate(gdim)

        gmsh_model_rank = 0
        mesh_comm = MPI.COMM_WORLD
        domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)
        return domain
    
    def create_vector(self, name, space, inter_cond) -> fem.Function:
        vec = fem.Function(space)
        vec.name = name
        vec.interpolate(inter_cond)
        return vec

    def boundary_condition(self) -> fem.dirichletbc:
        fdim = self.domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(
            self.domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
        bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(self.V, fdim, boundary_facets), self.V)
        return bc
    
    def create_solver(self, A):
        solver = PETSc.KSP().create(self.domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
        return solver

    def solve(self):
        domain = self.domain
        def velocity_field(x):
            return np.array([-2*np.pi*x[1], 2*np.pi*x[0]])
        
        u_n = self.create_vector(name='u_n', space=self.V, inter_cond=self.init_cond)
        w = self.create_vector(name='w', space=self.W, inter_cond=velocity_field)
        uh = self.create_vector(name='uh', space=self.V, inter_cond=self.init_cond)

        w_values = w.x.array.reshape((-1, domain.geometry.dim))
        w_inf_norm = np.linalg.norm(w_values, ord=np.inf)

        CFL = 0.5
        t = 0  # Start time
        T = self.T  # Final time
        dt = CFL*self.hmax/w_inf_norm
        num_steps = int(np.ceil(T/dt))

        bc = self.boundary_condition() 

        # Time-dependent output
        xdmf = io.XDMFFile(domain.comm, "linear_advection.xdmf", "w")
        xdmf.write_mesh(domain)

        # Variational problem and solver
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
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

        solver = self.create_solver(A)

        # Updating the solution and rhs per time step
        for _ in tqdm(range(num_steps)):
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
            xdmf.write_function(uh, t)

        xdmf.close()
        self.uh = uh

    def calc_error(self):
        """ Compute L2 error and error at nodes """
        u_ex = self.create_vector(name='u_ex', space=self.V, inter_cond=self.init_cond)
        error_L2 = np.sqrt(self.domain.comm.allreduce(fem.assemble_scalar(fem.form((self.uh - u_ex)**2 * ufl.dx)), op=MPI.SUM)) 
        return error_L2