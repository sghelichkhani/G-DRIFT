from gadopt import *
import gdrift
import numpy as np


# Set up geometry:
rmin, rmax, ref_level, nlayers = 1.208, 2.208, 4, 8

# Construct a CubedSphere mesh and then extrude into a sphere - note that unlike cylindrical case, popping is done internally here:
mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)
mesh = ExtrudedMesh(
    mesh2d,
    layers=nlayers,
    extrusion_type="radial"
)
# making sure we know mesh is spherical
mesh.cartesian = False

# Set up function spaces for coordinates
V = VectorFunctionSpace(mesh, "CG", 1)  # Temperature function space (scalar)
r = SpatialCoordinate(mesh)
v = Function(V, name="coordinates").interpolate(r / rmax * gdrift.R_earth)

# Set up scalar function spaces for seismic model fields
Q = FunctionSpace(mesh, "CG", 1)
vsh = Function(Q, name="vsh")
vsv = Function(Q, name="vsv")
v_ave = Function(Q, name="v_ave")

# Load the REVEAL model
seismic_model = gdrift.SeismicModel("REVEAL")

# Filling the vsh and vsv fields with the values from the seismic model
vsh.dat.data_with_halos[:] = np.squeeze(seismic_model.at(label="vsh", coordinates=v.dat.data_with_halos))
vsv.dat.data_with_halos[:] = np.squeeze(seismic_model.at(label="vsv", coordinates=v.dat.data_with_halos))

# Compute average of the layer
averager = LayerAveraging(mesh, quad_degree=6)
averager.extrapolate_layer_average(v_ave, averager.get_layer_average(vsh))


vtk_file = VTKFile("REVEAL.pvd")
vtk_file.write(vsh, vsv, v_ave)

# # Get the value of the vpv quantity at the CMB and the surface
# test_val = seismic_model.at(
#     label="vpv",
#     coordinates=np.array([[6370e3, 0, 0], [630e3, 0, 0]])
# )
