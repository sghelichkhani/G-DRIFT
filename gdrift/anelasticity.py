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

    def compute_Q_shear(self, depths, temperatures):
        """
        Computes the shear Q (matrix) based on depths and temperatures.

        Args:
            depths (numpy.ndarray): Depths at which the Q values are required.
            temperatures (numpy.ndarray): Temperatures corresponding to each depth.

        Returns:
            numpy.ndarray: A matrix of computed Q values.
        """
        depths = numpy.asarray(depths)
        temperatures = numpy.asarray(temperatures)

        Q_values = (self.B(depths) * (self.omega(depths)**self.a(depths))
                    * numpy.exp(
                        (self.a(depths) * self.g(depths)
                         * self.solidus.at_depth(depths)) / temperatures)
                    )
        return Q_values

    def compute_Q_bulk(self, depths, temperatures):
        """

        Args:
            depths (_type_): _description_
            temperatures (_type_): _description_
        """


def apply_anelastic_correction(thermo_model, anelastic_model):
    """
    Apply anelastic corrections to seismic velocity data using provided anelasticity model.
    within the low attenuation limit. For corrections we use the equation:
        $1 - \frac{V(anelastic)}{V(elastic)} = \frac{1}{2} cot(\frac{\alpha \pi}{2}) Q^{-1}$
    as described by Stixrude & Lithgow-Bertelloni 2005 JGR (doi:10.1029/2004JB002965, see equation 10)

    Args:
        thermo_model (ThermodynamicModel): The thermodynamic model with temperature and depth data.
        anelastic_model (BaseAnelasticityModel): An anelasticity model to compute Q values.

    Returns:
        Corrected ThermodynamicModel with anelastic effects.
    """
    class ThermodynamicModelPrime(thermo_model.__class__):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compute_swave_speed(self):
            """_summary_

            Returns:
                _type_: _description_
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
                swave_speed_table.get_vals()
                * (1 - 0.5/numpy.tan(anelastic_model.a(depths_x) * numpy.pi / 2) / Q_matrix)
            )

            # ) * (1 - (F / (numpy.pi * anelastic_model.a(depths_x))) * 1/Q_shear_matrix)
            return type(swave_speed_table)(
                x=swave_speed_table.get_x(),
                y=swave_speed_table.get_y(),
                vals=corrected_vals,
                name=f"{swave_speed_table.get_name()}_anelastically_corrected"
            )

        def compute_pwave_speed(self):
            """replacing the original compute_pwave_speed function
            by calculating the anelastic effect on compressional wave speed

            For quality factor of the p wave speed we refer to
            Don L. Anderson & R. S. Hart 1978 PEPI eq (1-3)
                $$Q_s = Q_{\mu}$$
                $$\farc{1}{Q_p} = \frac{L}{Q_\mu} + \frac{(1-L)}{Q_K}$$
                $$Q_K = \frac{(1-L) Q_\mu}{Q_s / Q_p - 1}$$
            with $L = 4/3 * \frac{\beta}{\alpha}$ where $\beta$ and $\alpha$
            are the compressional and shear wave velocities respectively.
            """
            # compute s and p wave velocities to compute "L".
            pwave_speed_table = super().compute_pwave_speed()
            swave_speed_table = super().compute_swave_speed()

            # Compute L
            L = 4 / 3 * (pwave_speed_table/swave_speed_table) ** 2

            # Meshing depths and temperatures of the anharmonic model to get all combinations
            depths_x, temperatures_x = numpy.meshgrid(
                pwave_speed_table.get_x(), pwave_speed_table.get_y(), indexing="ij")

            # computing Q_matrix for compressional wave
            Q_matrix = (
                L / anelastic_model.compute_Q_shear(depths_x, temperatures_x)
                + (1-L) / anelastic_model.compute_Q_bulk(depths_x, temperatures_x)
            )

            # Apply anelastic correction
            corrected_vals = (
                pwave_speed_table.get_vals() *
                (1 - 0.5/numpy.tan(anelastic_model.a(depths_x) * numpy.pi / 2) / Q_matrix)
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