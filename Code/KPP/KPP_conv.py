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
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver

from Utils.PDE_plot import PDE_plot
from Utils.RV import RV
from Utils.SI import SI
from Utils.helpers import get_nodal_h

from dolfinx.io import XDMFFile

script_dir = os.path.dirname(os.path.abspath(__file__))

location_fig = os.path.join(script_dir, 'Figures/RV') # location = './Figures'
location_data = os.path.join(script_dir, 'Data') # location = './Figures'

xdmf_file = location_data + '/KPP_exact.xdmf'

with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
