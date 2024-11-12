""" Seems to be the most correct one so far! """
""" TODO: Apply BC. Check that normalization of residual is correct, looks wrong """
import ufl
import numpy as np
from dolfinx import fem, mesh, io, plot
from petsc4py import PETSc
from mpi4py import MPI
import gmsh
from dolfinx.io import gmshio
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
import pyvista
import matplotlib as mpl

PLOT = True
# Domain and mesh parameters
gmsh.initialize()

membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
gmsh.model.occ.synchronize()

gdim = 2
gmsh.model.addPhysicalGroup(gdim, [membrane], 1)

hmax = 1/16 # 0.05 in example
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmax)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)
gmsh.model.mesh.generate(gdim)

gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

# Define function spaces
V = fem.functionspace(domain, ("Lagrange", 1))
W = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Define initial conditions
def initial_condition(x, r0=0.25, x0_1=0.3, x0_2=0):
    return (x[0] - x0_1)**2 + (x[1] - x0_2)**2 <= r0**2

def velocity_field(x):
    return np.array([-2*np.pi*x[1], 2*np.pi*x[0]])

u0 = fem.Function(V)
u0.name = "u0"
u0.interpolate(initial_condition)

# Define velocity field
w = fem.Function(W)
w.interpolate(velocity_field)

# Time-stepping parameters
CFL, Cvel, CRV = 0.5, 0.25, 1.0
w_inf_norm = np.max(np.sqrt(w.x.array[:].reshape(-1, 2)**2).sum(axis=1))
dt = CFL * hmax / w_inf_norm
T, t = 1.0, 0.0

# Set up mass and convection forms
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
mass_form = u * v * ufl.dx
conv_form = ufl.dot(w, ufl.grad(u)) * v * ufl.dx

M = assemble_matrix(fem.form(mass_form))
M.assemble()
C = assemble_matrix(fem.form(conv_form))
C.assemble()

# Time-stepping setup
xi_prev = fem.Function(V)
xi_prev.x.array[:] = u0.x.array[:]

xi = fem.Function(V)
xi.x.array[:] = u0.x.array[:]

solver = PETSc.KSP().create(domain.comm)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

if PLOT:
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

    plotter = pyvista.Plotter()
    plotter.open_gif("RV.gif", fps=10)

    grid.point_data["xi"] = xi.x.array
    warped = grid.warp_by_scalar("xi", factor=1)

    viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
    sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                position_x=0.1, position_y=0.8, width=0.8, height=0.1)

    renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
                                cmap=viridis, scalar_bar_args=sargs,
                                clim=[0, max(xi.x.array)])

while t < T:
    # Calculate residual R = M\(1/dt*M*xi - 1/dt*M*xi_prev + C*xi)
    # R = (1/dt) * (M @ xi.x.petsc_vec - M @ xi_prev.x.petsc_vec) + C @ xi.x.petsc_vec
    # R /= np.max(xi.x.array[:] - np.mean(xi.x.array[:]))  # Normalize
    temp_xi = PETSc.Vec().createWithArray(xi.x.array, comm=domain.comm)
    temp_xi_prev = PETSc.Vec().createWithArray(xi_prev.x.array, comm=domain.comm)

    M_xi = M.createVecRight()
    M.mult(temp_xi, M_xi)
    M_xi.scale(2 / dt)

    M_xi_prev = M.createVecRight()
    M.mult(temp_xi_prev, M_xi_prev)
    M_xi_prev.scale(1 / dt)

    C_xi = C.createVecRight()
    C.mult(temp_xi, C_xi)

    # Compute the right-hand side vector
    R = M_xi - M_xi_prev + C_xi
    R_norm = np.max(np.abs(R.array - np.mean(R.array)))
    # R.array[:] /= R_norm if R_norm != 0 else 1  # Normalize

    # Compute element-wise artificial viscosity
    Rv = fem.Function(V)
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    for cell in range(num_cells):
        # Per-element residual, velocity, and mesh size calculations
        loc2glb = V.dofmap.cell_dofs(cell)
        Rk = np.max(np.abs(R[loc2glb]))
        w_values = w.x.array.reshape((-1, domain.geometry.dim))
        Bk = np.max(np.sqrt(np.sum(w_values[loc2glb]**2, axis=1)))

        # Compute element size hk
        x = V.tabulate_dof_coordinates()[loc2glb]
        edges = [np.linalg.norm(x[i] - x[j]) for i in range(3) for j in range(i+1, 3)]
        hk = max(edges)
        
        # Element-wise viscosity
        epsilon_k = min(Cvel * hk * Bk, CRV * hk**2 * Rk)
        for dof in loc2glb:
            Rv.x.array[dof] += epsilon_k

    # Assemble system matrix A
    A_form = fem.form(2/dt * u * v * ufl.dx + ufl.dot(w, ufl.grad(u)) * v * ufl.dx + Rv * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    A = fem.petsc.assemble_matrix(A_form)
    A.assemble()

    # Assemble the residual-based viscosity term as a separate vector
    Rv_form = fem.form(Rv * ufl.inner(ufl.grad(xi), ufl.grad(v)) * ufl.dx)
    Rv_vec = fem.petsc.assemble_vector(Rv_form)

    # Construct the right-hand side vector
    L_vec = M_xi - C_xi - Rv_vec

    # Solve system A * xi = b
    solver.setOperators(A)
    solver.solve(L_vec, xi.x.petsc_vec)
    xi_prev.x.array[:] = xi.x.array[:]
    t += dt
    
    if PLOT:
        new_warped = grid.warp_by_scalar("xi", factor=1)
        warped.points[:, :] = new_warped.points
        warped.point_data["xi"][:] = xi.x.array
        plotter.write_frame()

if PLOT:
    plotter.close()