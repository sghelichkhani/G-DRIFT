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
from gdrift.profile import SplineProfile
import numpy as np

# Set up geometry:
rmin, rmax, ref_level, nlayers = 1.208, 2.208, 3, 4

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
X = SpatialCoordinate(mesh)
r = Function(V, name="coordinates").interpolate(X / rmax * gdrift.R_earth)


# Set up scalar function spaces for seismic model fields
Q = FunctionSpace(mesh, "CG", 1)
vsh = Function(Q, name="vsh")
vsv = Function(Q, name="vsv")
vs = Function(Q, name="vs")
v_ave = Function(Q, name="v_ave")

# Compute the depth field
depth = Function(Q, name="depth").interpolate(
    Constant(gdrift.R_earth) - sqrt(r[0]**(2) + r[1]**(2) + r[2]**(2))
)

# Load the REVEAL model
seismic_model = gdrift.SeismicModel("REVEAL")

# Filling the vsh and vsv fields with the values from the seismic model
reveal_vsh_vsv = seismic_model.at(label=["vsh", "vsv"], coordinates=r.dat.data_with_halos)
vsh.dat.data_with_halos[:] = reveal_vsh_vsv[:, 0]
vsv.dat.data_with_halos[:] = reveal_vsh_vsv[:, 1]

# Compute the isotropic velocity field
vs.interpolate(sqrt((2 * vsh ** 2 + vsv ** 2) / 3))

# Average the isotropic velocity field over the layers, this will be useful for visualising devaitons from the average
averager = LayerAveraging(mesh, quad_degree=6)
averager.extrapolate_layer_average(v_ave, averager.get_layer_average(vs))

# Compute a solidus for building anelasticity correction


def build_solidus():
    # Defining the solidus curve for manlte
    andrault_solidus = gdrift.RadialEarthModelFromFile(
        model_name="1d_solidus_Andrault_et_al_2011_EPSL",
        description="Andrault et al 2011 EPSL")

    # Defining parameters for Cammarano style anelasticity model
    hirsch_solidus = gdrift.HirschmannSolidus()

    my_depths = []
    my_solidus = []
    for solidus_model in [hirsch_solidus, andrault_solidus]:
        d_min, d_max = solidus_model.min_max_depth()
        dpths = np.arange(d_min, d_max, 10e3)
        my_depths.extend(dpths)
        my_solidus.extend(solidus_model.at_depth(dpths))

    my_depths.extend([3000e3])
    my_solidus.extend([solidus_model.at_depth(dpths[-1])])

    ghelichkhan_et_al = SplineProfile(
        depth=np.asarray(my_depths),
        value=np.asarray(my_solidus),
        name="Ghelichkhan et al 2021")

    return ghelichkhan_et_al


def build_anelasticity_model(solidus):
    def B(x):
        return np.where(x < 660e3, 1.1, 20)

    def g(x):
        return np.where(x < 660e3, 20, 10)

    def a(x):
        return 0.2

    def omega(x):
        return 1.0

    return gdrift.CammaranoAnelasticityModel(B, g, a, solidus, omega)


# Load PREM
prem = gdrift.PreliminaryRefEarthModel()

# Thermodynamic model
slb_pyrolite = gdrift.ThermodynamicModel("SLB_16", "pyrolite")
pyrolite_elastic_speed = slb_pyrolite.compute_swave_speed()

# building solidus model
solidus_ghelichkhan = build_solidus()
anelasticity = build_anelasticity_model(solidus_ghelichkhan)
anelastic_slb_pyrolite = gdrift.apply_anelastic_correction(
    slb_pyrolite, anelasticity)

# Define a field on Q for temperature
temperature = Function(Q, name="temperature")
t_ave = Function(Q, name="average_temperature")

# Convert the shear wave speed to temperature
temperature.dat.data_with_halos[:] = anelastic_slb_pyrolite.vs_to_temperature(
    vs.dat.data_with_halos,
    depth.dat.data_with_halos)

# Compute the layer-averaged temperature
averager.extrapolate_layer_average(t_ave, averager.get_layer_average(temperature))

# Write the fields to a VTK file
vtk_file = VTKFile("REVEAL.pvd")
vtk_file.write(vs, vsh, vsv, v_ave, temperature, t_ave)
