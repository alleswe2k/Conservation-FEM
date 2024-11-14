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
    def __init__(self):
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

    def boundary_condition(self, domain, V) -> fem.dirichletbc:
        fdim = domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
        bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)
        return bc
    
    def create_solver_linear(self, domain, A):
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
        return solver
    
    def plot(self, uh, V):
        # TODO: Fixa denna
        pyvista.start_xvfb()

        # Extract topology from mesh and create pyvista mesh
        topology, cell_types, x = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(topology, cell_types, x)

        # Set deflection values and add it to plotter
        grid.point_data["u"] = uh.x.array
        warped = grid.warp_by_scalar("u", factor=25)

        plotter = pyvista.Plotter()
        plotter.add_mesh(warped, show_edges=True, show_scalar_bar=True, scalars="u")
        new_warped = grid.warp_by_scalar("uh", factor=1)
        warped.points[:, :] = new_warped.points
        warped.point_data["uh"][:] = uh.x.array
        plotter.write_frame()
