import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyvista as pv
import ufl
import numpy as np

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

    def create_mesh_unit_disk(self, hmax):
        """ Creates a unit disk mesh given a hmax value """
        gdim = 2
        if self.initialized:
            gmsh.finalize()
        self.initialized = True
        gmsh.initialize()
        membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(gdim, [membrane], 1)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmax)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)
        gmsh.model.mesh.generate(gdim)

        gmsh_model_rank = 0
        mesh_comm = MPI.COMM_WORLD
        domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)
        return domain
    
    def create_vector(self, space, name, interpolation) -> fem.Function:
        """ Creates a fem.Function vector given a space and interpolation """
        vec = fem.Function(space)
        vec.name = name
        vec.interpolate(interpolation)
        return vec

    def boundary_condition(self, domain, V) -> fem.dirichletbc:
        """ Creates bc for the domain """
        fdim = domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
        bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)
        return bc
    
    def create_solver_linear(self, domain, A):
        """ Creates a linear solver """
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
        return solver
    
    def get_time_steps(self, domain, w, CFL, T, hmax):
        """ Get dt and num_steps for a given hmax and CFL """
        w_values = w.x.array.reshape((-1, domain.geometry.dim))
        w_inf_norm = np.linalg.norm(w_values, ord=np.inf)
        dt = CFL * hmax / w_inf_norm
        num_steps = int(np.ceil(T/dt))
        return dt, num_steps
    
    def get_patches(self, domain, V):
        """ Get a dictionary of the patch of every node in the domain """
        node_patches = {}
        # Loop over each cell to build node adjacency information
        for cell in range(domain.topology.index_map(domain.topology.dim).size_local):
            cell_nodes = V.dofmap.cell_dofs(cell)
            for node in cell_nodes:
                if node not in node_patches:
                    node_patches[node] = set()
                # Add all other nodes in this cell to the patch of the current node
                # node_patches[node].update(n for n in cell_nodes if n != node)
                node_patches[node].update(n for n in cell_nodes)
        return node_patches

    def __setup_plot(self, domain):
        tdim = domain.topology.dim
        os.environ["PYVISTA_OFF_SCREEN"] = "True"
        pv.start_xvfb()
        plotter = pv.Plotter(off_screen=True)
        domain.topology.create_connectivity(tdim, tdim)
        topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)

        return plotter, grid

    def plot_solution(self, domain, hmax, vector, title, filename, location="Figures"):
        """ Plots and saves the solution """
        pv.global_theme.colorbar_orientation = 'horizontal'
        plotter, grid = self.__setup_plot(domain)
        grid.point_data[title] = vector.x.array
        warped = grid.warp_by_scalar(title, factor=1)
        # Chooses the colormap
        viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
        sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                position_x=0.1, position_y=0.8, width=0.8, height=0.1)
        plotter.add_mesh(warped, show_edges=True, lighting=False,
                                cmap=viridis, scalar_bar_args=sargs,
                                clim=[0, max(vector.x.array)])

        # Take a screenshot
        plotter.screenshot(f"{location}/{filename}_{hmax}.png")  # Saves the plot as a PNG file
    

    def plot_2d(self, domain, hmax, vector, title, filename, location="Figures"):

        pv.global_theme.colorbar_orientation = 'vertical'
        plotter, grid = self.__setup_plot(domain)       
        grid.point_data[title] = vector.x.array
        warped = grid.warp_by_scalar(title, factor=1)

        # Chooses the colormap
        viridis = mpl.colormaps.get_cmap("viridis").resampled(25)

        sargs = {
            "title": title,
            "title_font_size": 20,
            "label_font_size": 15,
            "fmt": "%.2e",
            "color": "black",
            "position_x": 0.85,  # Position to the far right of the plot
            "position_y": 0.25,  # Center vertically
            "width": 0.08,  # Narrow width
            "height": 0.6  # Height proportional to the plot
        }

        plotter.add_mesh(
            warped,
            show_edges=False, 
            lighting=False,
            cmap=viridis,
            scalar_bar_args=sargs,
            clim=[min(vector.x.array), max(vector.x.array)])
        plotter.view_xy()
        # Take a screenshot
        plotter.screenshot(f"{location}/{filename}_{hmax}.png")  # Saves the plot as a PNG file




