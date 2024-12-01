import matplotlib.pyplot as plt
import numpy as np
import gdrift
from gdrift.profile import RadialProfileSpline

# In this tutorial we show how with the given functionalities
# we can apply anelastic correction to an existing thermodynamic table

# For our anelastic model in this example we use the study by Cammerano et al.
# The anelastic parameterisation is dependent on a solidus curve of the mantle.
# To this end we use the two datasets by Andrault et al 2011 EPSL
# And a solidus by a study by Hirschmann to build a combined solidus curve.


def build_solidus():
    # Defining the solidus curve for manlte
    # Andrault et al 2011
    andrault_solidus = gdrift.SolidusProfileFromFile(
        model_name="1d_solidus_Andrault_et_al_2011_EPSL",
        description="Andrault et al 2011 EPSL")

    # Hirsch solidus
    hirsch_solidus = gdrift.HirschmannSolidus()

    # Combining the two
    my_depths = []
    my_solidus = []
    for solidus_model in [hirsch_solidus, andrault_solidus]:
        d_min, d_max = solidus_model.min_max_depth()
        dpths = np.arange(d_min, d_max, 10e3)
        my_depths.extend(dpths)
        my_solidus.extend(solidus_model.at_depth(dpths))

    # Avoding unnecessary extrapolation by setting the solidus temperature at maximum depth
    my_depths.extend([3000e3])
    my_solidus.extend([solidus_model.at_depth(dpths[-1])])

    # building the solidu profile that was originally used by Ghelichkhan et al in their study
    ghelichkhan_et_al = RadialProfileSpline(
        depth=np.asarray(my_depths),
        value=np.asarray(my_solidus),
        name="Ghelichkhan et al 2021")

    return ghelichkhan_et_al


# Now the anelasticiy model by Cammarano et al can be constructed.
# Here we use B g a for making Q_\mu. and we provide a rough Q_\kappa
# Provided by Goes et al.

def build_anelasticity_model(solidus, N=3):
    N = str(N)
    Q = {
        "1": {
            "B": [0.5, 10],
            "g": [20, 10]
        },
        "2": {
            "B": [0.8, 15],
            "g": [20, 10]
        },
        "3": {
            "B": [1.1, 20],
            "g": [20, 10]
        },
        "4": {
            "B": [0.035, 2.25],
            "g": [30, 15]
        },
        "5": {
            "B": [0.056, 3.6],
            "g": [30, 15]
        },
        "6": {
            "B": [0.077, 4.95],
            "g": [30, 15]
        }
    }

    B = lambda x: np.where(x < 660e3, Q[N]["B"][0], Q[N]["B"][1])

    g = lambda x: np.where(x < 660e3, Q[N]["g"][0], Q[N]["g"][1])

    a = lambda x: 0.2

    omega = lambda x: 1.

    Q_kappa = lambda x: np.where(x < 660e3, 1e3, 1e4)

    return gdrift.CammaranoAnelasticityModel(B=B, g=g, a=a, solidus=solidus, Q_bulk=Q_kappa, omega=omega)


# Load PREM
prem = gdrift.PreliminaryRefEarthModel()

# Thermodynamic model
slb_pyrolite = gdrift.ThermodynamicModel("SLB_16", "pyrolite", temps=np.linspace(300, 4000), depths=np.linspace(0, 2890e3))
pyrolite_elastic_s_speed = slb_pyrolite.compute_swave_speed()
pyrolite_elastic_p_speed = slb_pyrolite.compute_pwave_speed()

# building solidus model
solidus_ghelichkhan = build_solidus()
N = 6 # choose model from cammarano et al., 2003
anelasticity = build_anelasticity_model(solidus_ghelichkhan, 1)
anelastic_slb_pyrolite = gdrift.apply_anelastic_correction(slb_pyrolite, anelasticity)
pyrolite_anelastic_s_speed = anelastic_slb_pyrolite.compute_swave_speed()
pyrolite_anelastic_p_speed = anelastic_slb_pyrolite.compute_pwave_speed()

# contour lines to plot
cntr_lines = np.linspace(4000, 7000, 20)

plt.close("all")
fig, axes = plt.subplots(ncols=2)
axes[0].set_position([0.1, 0.1, 0.35, 0.8])
axes[1].set_position([0.5, 0.1, 0.35, 0.8])
# Getting the coordinates
depths_x, temperatures_x = np.meshgrid(
    slb_pyrolite.get_depths(), slb_pyrolite.get_temperatures(), indexing="ij")
img = []

for id, table in enumerate([pyrolite_elastic_s_speed, pyrolite_anelastic_s_speed]):
    img.append(axes[id].contourf(
        temperatures_x,
        depths_x,
        table.get_vals(),
        cntr_lines,
        cmap=plt.colormaps["autumn"].resampled(20),
        extend="both"))
    axes[id].invert_yaxis()
    axes[id].set_xlabel("Temperature [K]")
    axes[id].set_ylabel("Depth [m]")
    axes[id].grid()

axes[1].set_ylabel("")
axes[1].set_yticklabels("")

axes[0].text(0.5, 1.05, s="Elastic", transform=axes[0].transAxes,
             ha="center", va="center",
             bbox=dict(facecolor=(1.0, 1.0, 0.7)))
axes[1].text(0.5, 1.05, s="With Anelastic Correction",
             ha="center", va="center",
             transform=axes[1].transAxes, bbox=dict(facecolor=(1.0, 1.0, 0.7)))
fig.colorbar(img[-1], ax=axes[0], cax=fig.add_axes([0.88,
             0.1, 0.02, 0.8]), orientation="vertical", label="Shear-Wave Speed [m/s]")


# Figure 2:
# Looking at a specific depth of shear seismic speed
plt.close(2)
fig_2 = plt.figure(num=2)
ax_2 = fig_2.add_subplot(111)
index = 100
ax_2.plot(pyrolite_anelastic_s_speed.get_y(),
          pyrolite_anelastic_s_speed.get_vals()[index, :], color="blue", label="With Anelastic Correction")
ax_2.plot(pyrolite_anelastic_s_speed.get_y(),
          pyrolite_elastic_s_speed.get_vals()[index, :], color="red", label="Elastic Model")
ax_2.vlines(
    [solidus_ghelichkhan.at_depth(pyrolite_anelastic_s_speed.get_x()[index])],
    ymin=pyrolite_anelastic_s_speed.get_vals()[index, :].min(),
    ymax=pyrolite_anelastic_s_speed.get_vals()[index, :].max(),
    color="grey", label="Solidus", alpha=0.5)

ax_2.set_xlabel("Temperature[K]")
ax_2.set_ylabel("Shear Seismic-Wave Speed [m/s]")
ax_2.text(
    0.5, 1.05, s=f"cammarano et al. Q{N} at depth {pyrolite_anelastic_s_speed.get_x()[index] / 1e3:.1f} [km]",
    ha="center", va="center",
    transform=ax_2.transAxes, bbox=dict(facecolor=(1.0, 1.0, 0.7)))
ax_2.legend()
ax_2.grid()
plt.show()


# Figure 3:
# Looking at a specific depth of compressional seismic speed
plt.close(3)
fig_3 = plt.figure(num=3)
ax_3 = fig_3.add_subplot(111)
index = 100
ax_3.plot(pyrolite_anelastic_p_speed.get_y(),
          pyrolite_anelastic_p_speed.get_vals()[index, :], color="blue", label="With Anelastic Correction")
ax_3.plot(pyrolite_anelastic_p_speed.get_y(),
          pyrolite_elastic_p_speed.get_vals()[index, :], color="red", label="Elastic Model")
ax_3.vlines(
    [solidus_ghelichkhan.at_depth(pyrolite_anelastic_p_speed.get_x()[index])],
    ymin=pyrolite_anelastic_p_speed.get_vals()[index, :].min(),
    ymax=pyrolite_anelastic_p_speed.get_vals()[index, :].max(),
    color="grey", label="Solidus", alpha=0.5)

ax_3.set_xlabel("Temperature[K]")
ax_3.set_ylabel("Compressional Seismic-Wave Speed [m/s]")
ax_3.text(
    0.5, 1.05, s=f"cammarano et al. Q{N} at depth {pyrolite_anelastic_p_speed.get_x()[index] / 1e3:.1f} [km]",
    ha="center", va="center",
    transform=ax_3.transAxes, bbox=dict(facecolor=(1.0, 1.0, 0.7)))
ax_3.legend()
ax_3.grid()
plt.show()
