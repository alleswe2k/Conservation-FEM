import os

import matplotlib as mpl
import matplotlib.pyplot as plt

import pyvista as pv
import numpy as np

from dolfinx import plot

class PDE_plot():
    def __init__(self):
        self.started_anim = False

    def __setup_plot(self, domain, vector, title):
        tdim = domain.topology.dim
        os.environ["PYVISTA_OFF_SCREEN"] = "True"
        pv.start_xvfb()
        plotter = pv.Plotter(off_screen=True)
        domain.topology.create_connectivity(tdim, tdim)
        topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)

        grid.point_data[title] = vector.x.array

        desired_max_height = 1
        scalar_field = grid[title]
        scalar_range = scalar_field.max() - scalar_field.min()

        if scalar_range > 0:
            scale_factor = desired_max_height / scalar_range
        else:
            scale_factor = 1.0

        # Apply the warping with the calculated scale factor
        warped = grid.warp_by_scalar(title, factor=scale_factor)

        color_map = mpl.colormaps.get_cmap("viridis").resampled(25)

        return plotter, warped, color_map


    def plot_pv_3d(self, domain, mesh_size, vector, title, filename, location=""):
        pv.global_theme.colorbar_orientation = 'horizontal'
        plotter, warped, color_map = self.__setup_plot(domain, vector, title)

        sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                position_x=0.1, position_y=0.8, width=0.8, height=0.1)
        plotter.add_mesh(warped, show_edges=False, lighting=False,
                                cmap=color_map, scalar_bar_args=sargs,
                                clim=[min(vector.x.array), max(vector.x.array)])

        # Take a screenshot
        plotter.screenshot(f"{location}/{filename}_{mesh_size}.png")  # Saves the plot as a PNG file
    

    def plot_pv_2d(self, domain, mesh_size, vector, title, filename, location=""):

        pv.global_theme.colorbar_orientation = 'vertical'
        plotter, warped, color_map = self.__setup_plot(domain, vector, title)       


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
            cmap=color_map,
            scalar_bar_args=sargs,
            clim=[min(vector.x.array), max(vector.x.array)])
        plotter.view_xy()
        # Take a screenshot
        plotter.screenshot(f"{location}/{filename}_{mesh_size}.png")  # Saves the plot as a PNG file


    def plot_convergence(self, L2_errors, title, filename, location="Figures"):
        x = np.array([4, 8, 16, 32])
        fit = np.polyfit(np.log10(x), np.log10(L2_errors), 1)
        y = 10**fit[1] * x**fit[0]

        # Plot L2 errors and fitted line
        plt.figure(figsize=(8, 6))
        plt.loglog(x, L2_errors, '-^', label="L2 Errors")
        plt.loglog(x, y, '--', label="Fitted Line")

        # Add a denser grid
        plt.grid(which='both', linestyle='--', linewidth=0.5)
        # Add a text below legend that displays the convergence
        legend_box = plt.legend()
        txt = mpl.offsetbox.TextArea(f"Convergence: {abs(fit[0]):.2f}")
        box = legend_box._legend_box
        box.get_children().append(txt)
        box.set_figure(box.figure)

        plt.xlabel(r"$1/h$")
        plt.ylabel(r"$||e||$")
        plt.title(f'Convergence for {title}')

        # Save the plot
        plt.savefig(f'{location}/{filename}.png')
        plt.show()
