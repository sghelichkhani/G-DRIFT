# This script demonstrates how to set up a spherical mesh using the `gadopt` and `gdrift` libraries,
# interpolate seismic model data onto the mesh, and compute layer-averaged seismic velocities.
# The resulting fields are then written to a VTK file for visualization.
#
# Steps:
# 1. Set up the geometry and construct a spherical mesh.
# 2. Define function spaces for coordinates and seismic model fields.
# 3. Load the REVEAL seismic model and interpolate its data onto the mesh.
# 4. Compute the layer-averaged seismic velocities.
# 5. Write the fields to a VTK file for visualization.
#
# Libraries used:
# - `gadopt`: For mesh generation and manipulation.
# - `gdrift`: For loading and handling seismic models.
# - `numpy`: For numerical operations.

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
vs = Function(Q, name="vs")
v_ave = Function(Q, name="v_ave")

# Load the REVEAL model
seismic_model = gdrift.SeismicModel("REVEAL")

# Filling the vsh and vsv fields with the values from the seismic model
vsh.dat.data_with_halos[:] = np.squeeze(seismic_model.at(label="vsh", coordinates=v.dat.data_with_halos))
vsv.dat.data_with_halos[:] = np.squeeze(seismic_model.at(label="vsv", coordinates=v.dat.data_with_halos))
vs.interpolate(sqrt((2 * vsh ** 2 + vsv ** 2) / 3))

# Compute average of the layer
averager = LayerAveraging(mesh, quad_degree=6)
averager.extrapolate_layer_average(v_ave, averager.get_layer_average(vs))

# Write the fields to a VTK file
vtk_file = VTKFile("REVEAL.pvd")
vtk_file.write(vs, vsh, vsv, v_ave)