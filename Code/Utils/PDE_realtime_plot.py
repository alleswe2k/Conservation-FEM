import matplotlib as mpl
import pyvista
import numpy as np

from dolfinx import plot

class PDE_realtime_plot:
    def __init__(self, location_figures, uh, epsilon, fs, numerator):
        self.PLOT_SOL = False
        self.uh = uh
        self.epsilon = epsilon
        self.numerator = numerator
        if self.PLOT_SOL:
            self.grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(fs))

            self.plotter = pyvista.Plotter()
            self.plotter.open_gif(f"{location_figures}/smoothness.gif", fps=10)

            self.grid.point_data["uh"] = uh.x.array
            self.warped = self.grid.warp_by_scalar("uh", factor=1)

            viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
            sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                        position_x=0.1, position_y=0.8, width=0.8, height=0.1)

            self.renderer = self.plotter.add_mesh(self.warped, show_edges=True, lighting=False,
                                        cmap=viridis, scalar_bar_args=sargs,
                                        clim=[0, max(uh.x.array)])
        else:
            # Set up PyVista multi-view self.plotter
            self.plotter = pyvista.Plotter(shape=(1, 2), off_screen=False, window_size=[2500, 1600])
            self.plotter.open_gif(f"{location_figures}/solution_and_epsilon.gif", fps=10)

            # Create grid for uh (solution)
            self.grid_uh = pyvista.UnstructuredGrid(*plot.vtk_mesh(fs))
            self.grid_uh.point_data["uh"] = self.uh.x.array

            # Configure left subplot for uh
            self.plotter.subplot(0, 0)  # Left subplot
            self.plotter.add_text("Solution (uh)", font_size=10, position="upper_left")
            viridis_uh = mpl.colormaps.get_cmap("viridis").resampled(25)
            self.plotter.add_mesh(self.grid_uh, show_edges=True, cmap=viridis_uh,
                            scalar_bar_args=dict(title="uh", title_font_size=12, label_font_size=10),
                            clim=[np.min(self.uh.x.array), np.max(self.uh.x.array)])
            self.plotter.view_xy()

            # Create grid for epsilon (artificial viscosity)
            self.grid_epsilon = pyvista.UnstructuredGrid(*plot.vtk_mesh(fs))
            self.grid_epsilon.point_data["epsilon"] = self.epsilon.x.array

            # Configure right subplot for epsilon
            self.plotter.subplot(0, 1)  # Right subplot
            self.plotter.add_text("Artificial Viscosity (epsilon)", font_size=10, position="upper_left")
            viridis_epsilon = mpl.colormaps.get_cmap("viridis").resampled(25)
            self.plotter.add_mesh(self.grid_epsilon, show_edges=True, cmap=viridis_epsilon,
                            scalar_bar_args=dict(title="epsilon", title_font_size=12, label_font_size=10),
                            clim=[np.min(self.epsilon.x.array), np.max(self.epsilon.x.array)])
            self.plotter.view_xy()

            # Create grid for numerator
            # self.grid_numerator = pyvista.UnstructuredGrid(*plot.vtk_mesh(fs))
            # self.grid_numerator.point_data["numerator"] = self.numerator.x.array

            # self.plotter.subplot(0, 1)  # Right subplot
            # self.plotter.add_text("Numerator", font_size=10, position="upper_left")
            # viridis_numerator = mpl.colormaps.get_cmap("viridis").resampled(25)
            # self.plotter.add_mesh(self.grid_numerator, show_edges=True, cmap=viridis_numerator,
            #                 scalar_bar_args=dict(title="numerator", title_font_size=12, label_font_size=10),
            #                 clim=[np.min(self.numerator.x.array), np.max(self.numerator.x.array)])
            # self.plotter.view_xy()

    
    def update_plot(self, uh, epsilon, numerator):
        self.uh = uh
        self.epsilon = epsilon
        self.numerator = numerator
        if self.PLOT_SOL:
            new_warped = self.grid.warp_by_scalar("uh", factor=1)
            self.warped.points[:, :] = new_warped.points
            self.warped.point_data["uh"][:] = self.uh.x.array
            self.plotter.write_frame()
        else:
            self.grid_uh.point_data["uh"][:] = self.uh.x.array
            self.plotter.subplot(0, 0)
            self.plotter.update_scalar_bar_range([np.min(self.uh.x.array), np.max(self.uh.x.array)])

            # Update epsilon data
            self.grid_epsilon.point_data["epsilon"][:] = self.epsilon.x.array
            self.plotter.subplot(0, 1)
            self.plotter.update_scalar_bar_range([np.min(self.epsilon.x.array), np.max(self.epsilon.x.array)])

            # self.grid_numerator.point_data["numerator"][:] = self.numerator.x.array
            # self.plotter.subplot(0, 1)
            # self.plotter.update_scalar_bar_range([np.min(self.numerator.x.array), np.max(self.numerator.x.array)])

            # Write the updated frame
            self.plotter.write_frame()
    
    def close(self):
        self.plotter.close()