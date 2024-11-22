from abc import ABC, abstractmethod
import numpy


class BaseAnelasticityModel(ABC):
    """
    Abstract base class for an anelasticity model.
    Abstract base class for an anelasticity model.
    All anelasticity models must be able to compute a Q matrix given depths and temperatures.
    """

    @abstractmethod
    def compute_Q_shear(self, depths, temperatures):
        """
        Computes the s-wave anelastic quality factor (Q) matrix for given depths and temperatures.

        Args:
            depths (numpy.ndarray): Array of depths at which Q values are required.
            temperatures (numpy.ndarray): Array of temperatures corresponding to the depths.

        Returns:
            numpy.ndarray: A matrix of Q values corresponding to the given depths and temperatures.
        """
        pass

    @abstractmethod
    def compute_Q_bulk(self, depths, temperatures):
        """
        Computes the compressional anelastic quality factor (Q) matrix for given depths and temperatures.

        Args:
            depths (numpy.ndarray): Array of depths at which Q values are required.
            temperatures (numpy.ndarray): Array of temperatures corresponding to the depths.

        Returns:
            numpy.ndarray: A matrix of Q values corresponding to the given depths and temperatures.
        """
        pass


class CammaranoAnelasticityModel(BaseAnelasticityModel):
    """
    A specific implementation of an anelasticity model following the approach by Cammarano et al.
    """

    def __init__(self, B, g, a, solidus, Q_bulk=lambda x: 10000, omega=lambda x: 1.0):
        """
        Initialize the model with the given parameters.

        Args:
            B (function): Scaling factor for the Q model.
            g (function): Activation energy parameter.
            a (function): Frequency dependency parameter.
            solidus (function): Solidus temperature for mantle.
            omega (function): Seismic frequency (default is 1).
        """
        self.B = B
        self.g = g
        self.a = a
        self.omega = omega
        self.solidus = solidus
        self.Q_bulk = Q_bulk

    def compute_Q_shear(self, depths, temperatures):
        """
        Compute the shear Q (attenuation quality factor) matrix based on input depths and temperatures.

        Args:
            depths (numpy.ndarray): An array of depths at which Q values are to be calculated.
            temperatures (numpy.ndarray): An array of temperatures corresponding to the specified depths.

        Returns:
            numpy.ndarray: A matrix of calculated Q values, representing the shear attenuation quality factor.

        Notes:
            - If either `depths` or `temperatures` has a single element, it will be broadcasted.
            - For a full Q_shear matrix, `depths` and `temperatures` should be of compatible shapes (e.g., generated via `numpy.meshgrid`).
            - The computation uses the properties of the material, such as `B`, `omega`, `a`, and `g`, along with the solidus temperature at a given depth.

        Example:
            # Example usage:
            depths = numpy.linspace(0, 100, 50)  # 50 depth points from 0 to 100 km
            # Corresponding temperatures
            temperatures = numpy.linspace(800, 1200, 50)
            Q_matrix = model.compute_Q_shear(depths, temperatures)
        """
        depths = numpy.asarray(depths)
        temperatures = numpy.asarray(temperatures)

        Q_values = (
            self.B(depths) * (self.omega(depths)**self.a(depths)) * numpy.exp((self.a(depths) * self.g(depths) * self.solidus(depths)) / temperatures)
        )

        return Q_values

    def compute_Q_bulk(self, depths, temperatures):
        """

        Args:
            depths (_type_): _description_
            temperatures (_type_): _description_
        """
        return self.Q_bulk(depths)


def apply_anelastic_correction(thermo_model, anelastic_model):
    """
    Apply anelastic corrections to seismic velocity data using the provided "anelastic_model"
    within the low attenuation limit. The corrections are based on the equation:
        $1 - \frac{V(anelastic)}{V(elastic)} = \frac{1}{2} \cot(\frac{\alpha \pi}{2}) Q^{-1}$
    as described by Stixrude & Lithgow-Bertelloni (doi:10.1029/2004JB002965, Eq-10).

        thermo_model (ThermodynamicModel): The thermodynamic model containing temperature and depth data.

        ThermodynamicModel: A new thermodynamic model with anelastically corrected seismic velocities.

    The returned model includes the following methods with anelastic corrections:

    - compute_swave_speed: Calculates the anelastic effect on shear wave speed.
    - compute_pwave_speed: Calculates the anelastic effect on compressional wave speed.

    The `compute_swave_speed` method applies the anelastic correction to the shear wave speed using the provided
    anelastic model. The `compute_pwave_speed` method applies the anelastic correction to the compressional wave speed
    using the quality factor derived from the equations provided by Don L. Anderson & R. S. Hart (1978, PEPI eq 1-3).

    The corrections are applied by meshing the depths and temperatures from the thermodynamic model and computing the
    quality factor matrices for shear and bulk moduli. The corrected seismic velocities are then calculated and returned
    as new tables with the corrected values.
    """
    class ThermodynamicModelPrime(thermo_model.__class__):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compute_swave_speed(self):
            """
            Computes the anelastically corrected shear wave speed.
            This method first retrieves the shear wave speed table from the superclass.
            It then creates a meshgrid of depths and temperatures based on the table's
            x and y values. Using these grids, it computes the shear wave quality factor
            (Q) matrix using the provided anelastic model. The shear wave speed values
            are then corrected for anelasticity using the computed Q matrix and the
            anelastic model's parameter 'a'. The corrected shear wave speed values are
            returned in a new table with the same x and y values but updated values and
            name.

            Returns:
                A table of anelastically corrected shear wave speed values.
            """
            #
            swave_speed_table = super().compute_swave_speed()
            depths_x, temperatures_x = numpy.meshgrid(
                swave_speed_table.get_x(),
                swave_speed_table.get_y(),
                indexing="ij")

            # For shear wave velocity Q = Q_\mu
            Q_matrix = anelastic_model.compute_Q_shear(
                depths_x, temperatures_x)

            # Anelastically corrected shear wave values
            corrected_vals = (
                swave_speed_table.get_vals() * (1 - 0.5 / numpy.tan(anelastic_model.a(depths_x) * numpy.pi / 2) / Q_matrix)
            )

            #
            return type(swave_speed_table)(
                x=swave_speed_table.get_x(),
                y=swave_speed_table.get_y(),
                vals=corrected_vals,
                name=f"{swave_speed_table.get_name()}_anelastically_corrected"
            )

        def compute_pwave_speed(self):
            """
            Calculate the anelastic effect on compressional wave speed.

            This method replaces the original `compute_pwave_speed` function by incorporating
            the anelastic effect on compressional wave speed. The quality factor for the
            P - wave speed is derived from the equations provided by Don L. Anderson & R. S. Hart
            (1978, PEPI eq 1 - 3):

                Q_s = Q_{\\mu}
                \frac{1}{Q_p} = \frac{L}{Q_\\mu} + \frac{(1-L)}{Q_K}
                Q_K = \frac{(1-L) Q_\\mu}{Q_s / Q_p - 1}

            where (L = \frac{4}{3} \\left( \frac{\beta}{\alpha} \right)^2 \\), and (\\beta)
            and (\\alpha) are the shear and compressional wave velocities, respectively.

            Returns:
                Table: Anelastically corrected compressional wave speed table.
            """            # compute s and p wave velocities to compute "L".
            pwave_speed_table = super().compute_pwave_speed()
            swave_speed_table = super().compute_swave_speed()

            # Compute L
            L = 4 / 3 * (pwave_speed_table.get_vals() / swave_speed_table.get_vals()) ** 2

            # Meshing depths and temperatures of the anharmonic model to get all combinations
            depths_x, temperatures_x = numpy.meshgrid(
                pwave_speed_table.get_x(), pwave_speed_table.get_y(), indexing="ij")

            # computing Q_matrix for compressional wave
            Q_matrix_inv = (
                L / anelastic_model.compute_Q_shear(depths_x, temperatures_x) + (1 - L) / anelastic_model.compute_Q_bulk(depths_x, temperatures_x)
            )

            # Apply anelastic correction
            corrected_vals = (
                pwave_speed_table.get_vals() * (1 - 0.5 / numpy.tan(anelastic_model.a(depths_x) * numpy.pi / 2) * Q_matrix_inv)
            )

            # return the anelastically corrected table
            return type(pwave_speed_table)(
                x=pwave_speed_table.get_x(),
                y=pwave_speed_table.get_y(),
                vals=corrected_vals,
                name=f"{pwave_speed_table.get_name()}_anelastically_corrected"
            )

    return ThermodynamicModelPrime(
        thermo_model.model,
        thermo_model.composition,
        thermo_model.get_temperatures(),
        thermo_model.get_depths()
    )
