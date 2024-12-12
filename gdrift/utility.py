import numpy
import scipy
from .constants import R_earth, R_cmb


def is_ascending(lst):
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))


def is_descending(lst):
    return all(lst[i] >= lst[i + 1] for i in range(len(lst) - 1))


def compute_mass(radius, density):
    """
    Compute the mass enclosed within each radius using the cumulative trapezoidal rule.

    Args:
        radius (numpy.ndarray): Array of radii from the center of the Earth or other celestial body.
        density (numpy.ndarray): Array of densities corresponding to each radius.

    Returns:
        numpy.ndarray: Array of cumulative mass enclosed up to each radius.
    """
    if radius[0] != 0:
        raise ValueError(
            f"The first element radius should be zero, but it is {radius[0]}")

    mass_enclosed = numpy.zeros_like(radius)
    for i in range(1, len(radius)):
        shell_volume = 4 / 3 * numpy.pi * (radius[i]**3 - radius[i - 1]**3)
        average_density = (density[i] + density[i - 1]) / 2
        mass_enclosed[i] = mass_enclosed[i - 1] + shell_volume * average_density
    return mass_enclosed


def compute_gravity(radius, mass_enclosed):
    """
    Compute gravitational acceleration at each radius based on the enclosed mass.

    Args:
        radius (numpy.ndarray): Array of radii from the center.
        mass_enclosed (numpy.ndarray): Array of cumulative mass enclosed up to each radius.

    Returns:
        numpy.ndarray: Array of gravitational acceleration at each radius.
    """
    gravity = numpy.zeros_like(radius)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        gravity = scipy.constants.G * mass_enclosed / radius**2
        # approximate central gravity as slightly above it to avoid NaN
        gravity[0] = gravity[1]
    return gravity


def compute_pressure(radius, density, gravity):
    """
    Calculate the hydrostatic pressure at each radius based on the density and gravitational acceleration.

    Args:
        radius (numpy.ndarray): Array of radii from the center to the surface.
        density (numpy.ndarray): Array of densities at each radius.
        gravity (numpy.ndarray): Array of gravitational accelerations at each radius.

    Returns:
        numpy.ndarray: Array of pressures calculated from the surface inward to each radius.
    """
    pressure = numpy.zeros_like(radius)
    for i in range(len(radius) - 2, -1, -1):
        dr = radius[i + 1] - radius[i]
        avg_density = (density[i] + density[i + 1]) / 2
        avg_gravity = (gravity[i] + gravity[i + 1]) / 2
        pressure[i] = pressure[i + 1] + avg_density * avg_gravity * dr
    return pressure


def geodetic_to_cartesian(lat, lon, depth, earth_radius=R_earth):
    """
    Convert geographic coordinates to Cartesian coordinates.

    Parameters:
    lat (float or numpy.ndarray): Latitude in degrees.
    lon (float or numpy.ndarray): Longitude in degrees.
    depth (float or numpy.ndarray): Depth below Earth's surface in km.
    earth_radius (float): Radius of the Earth in km. Default is 6371 km.

    Returns:
    tuple: Cartesian coordinates (x, y, z).
    """
    # Convert latitude and longitude from degrees to radians
    lat_rad = numpy.radians(lat)
    lon_rad = numpy.radians(lon)

    r = earth_radius - depth

    # Compute Cartesian coordinates
    x = r * numpy.cos(lat_rad) * numpy.cos(lon_rad)
    y = r * numpy.cos(lat_rad) * numpy.sin(lon_rad)
    z = r * numpy.sin(lat_rad)

    return x, y, z


def cartesian_to_geodetic(x, y, z, earth_radius=R_earth):
    """
    Convert Cartesian coordinates to geographic coordinates.

    Parameters:
    x (float or numpy.ndarray): x coordinate in km.
    y (float or numpy.ndarray): y coordinate in km.
    z (float or numpy.ndarray): z coordinate in km.
    earth_radius (float): Radius of the Earth in km. Default is 6371e3 m.

    Returns:
    tuple: Geographic coordinates (lat, lon, depth).
    """
    # Compute the distance from the Earth's center
    r = numpy.sqrt(x**2 + y**2 + z**2)

    # Compute latitude in radians
    lat_rad = numpy.arcsin(z / r)

    # Compute longitude in radians
    lon_rad = numpy.arctan2(y, x)

    # Compute depth below Earth's surface
    depth = earth_radius - r

    # Convert latitude and longitude from radians to degrees
    lat = numpy.degrees(lat_rad)
    lon = numpy.degrees(lon_rad)

    return lat, lon, depth


def cartesian_to_spherical(x, y, z):
    """
    Converts Cartesian coordinates to spherical coordinates.

    Parameters:
    x (float): x-coordinate in Cartesian coordinates.
    y (float): y-coordinate in Cartesian coordinates.
    z (float): z-coordinate in Cartesian coordinates.

    Returns:
    tuple: Spherical coordinates (r, theta, phi).
    """

    # Calculate the radial distance
    r = numpy.sqrt(x**2 + y**2 + z**2)

    # Calculate the polar angle (theta)
    theta = numpy.arccos(z / r)

    # Calculate the azimuthal angle (phi)
    phi = numpy.arctan2(y, x)

    return (r, theta, phi)


def spherical_to_cartesian(r, theta, phi):
    """
    Converts spherical coordinates to Cartesian coordinates.

    Parameters:
    r (float): Radial distance in spherical coordinates.
    theta (float): Polar angle in spherical coordinates.
    phi (float): Azimuthal angle in spherical coordinates.

    Returns:
    tuple: Cartesian coordinates (x, y, z).
    """
    # Calculate the x-coordinate
    x = r * numpy.sin(theta) * numpy.cos(phi)

    # Calculate the y-coordinate
    y = r * numpy.sin(theta) * numpy.sin(phi)

    # Calculate the z-coordinate
    z = r * numpy.cos(theta)

    return (x, y, z)


def nondimensionalise_coords(x, y, z, R_nd_earth=2.22, R_nd_cmb=1.22):
    r, theta, phi = cartesian_to_spherical(x, y, z)

    # Calculate the slope (a)
    a = (R_nd_earth - R_nd_cmb) / (R_earth - R_cmb)
    # Calculate the intercept (b)
    b = R_nd_earth - a * R_earth

    r_scaled = a * r + b
    x_prime, y_prime, z_prime = spherical_to_cartesian(r_scaled, theta, phi)
    return (x_prime, y_prime, z_prime)


def dimensionalise_coords(x, y, z, R_nd_cmb=1.22, R_nd_earth=2.22):
    """
    """
    r, theta, phi = cartesian_to_spherical(x, y, z)

    # Calculate the slope (a)
    a = (R_earth - R_cmb) / (R_nd_earth - R_nd_cmb)
    # Calculate the intercept (b)
    b = R_earth - a * R_nd_earth

    r_scaled = a * r + b
    x_prime, y_prime, z_prime = spherical_to_cartesian(r_scaled, theta, phi)

    return (x_prime, y_prime, z_prime)


def fibonacci_sphere(n):
    """Generates points on a sphere using the Fibonacci sphere algorithm, which
    distributes points **approximately** evenly over the surface of a sphere.

    This method calculates coordinates for each point using the golden angle,
    ensuring that each point is equidistant from its neighbors. The algorithm
    is particularly useful for creating evenly spaced points on a sphere's
    surface without clustering at the poles, a common issue in other spherical
    point distribution methods.

    Args:
        n (int): The number of points to generate on the sphere's surface.

    Returns:
        numpy.ndarray: A 2D array of shape (n, 3), where each row
                       contains the [x, y, z] coordinates of a point on the
                       sphere.

    Example:
        >>> sphere = _fibonacci_sphere(100)
        >>> print(sphere.shape)
        (100, 3)

    """

    phi = numpy.pi * (3. - numpy.sqrt(5.))  # golden angle in radians

    y = 1 - (numpy.arange(n) / (n - 1)) * 2
    radius = numpy.sqrt(1 - y * y)
    theta = phi * numpy.arange(n)
    x = numpy.cos(theta) * radius
    z = numpy.sin(theta) * radius
    return numpy.array([[x[i], y[i], z[i]] for i in range(len(x))])


def enlist(obj):
    """ Enlist makes sure we have a list

    Args:
        obj of any kind

    Returns:
        a list
    """
    if isinstance(obj, list):
        return obj
    else:
        return [obj]


def interpolate_to_points(values, distances, inds, min_distance=1e-6):
    """
    Interpolate field data to given query points using weighted averaging.

    Parameters
    ----------
    values : np.ndarray
        Array of shape (n, ...) containing the field data to interpolate from.
    distances : np.ndarray
        Array of shape (n, k) containing the distances to the k nearest neighbors.
    inds : np.ndarray
        Array of shape (n, k) containing the indices of the k nearest neighbors.
    min_distance : float, optional
        Minimum distance to avoid division by zero. Default is 1e-6.

    Returns
    -------
    np.ndarray
        Interpolated field data at the query points.
    """
    safe_dists = numpy.where(distances < min_distance, min_distance, distances)
    replace_flg = distances[:, 0] < min_distance

    if len(values.shape) > 1:
        weights = 1 / safe_dists
        weighted_sum = numpy.einsum("ij, ijk -> ik", weights, values[inds])
        ret = weighted_sum / numpy.sum(weights, axis=1)[:, numpy.newaxis]
        ret[replace_flg, :] = values[inds[replace_flg, 0], :]
    else:
        weights = 1 / safe_dists
        weighted_sum = numpy.einsum("ij, ij -> i", weights, values[inds])
        ret = weighted_sum / numpy.sum(weights, axis=1)
        ret[replace_flg] = values[inds[replace_flg, 0]]

    return ret


def create_labeled_array(data_dict, labels):
    """
    Create a labeled array from a dictionary of arrays and a list of labels.

    Args:
        data_dict (dict): Dictionary where keys are labels and values are arrays of length n.
        labels (list): List of strings representing the labels.

    Returns:
        numpy.ndarray: Array of shape (n, m) where m is the number of labels.
    """
    n = len(next(iter(data_dict.values())))
    m = len(labels)
    labeled_array = numpy.zeros((n, m))

    for i, label in enumerate(labels):
        if label in data_dict:
            labeled_array[:, i] = data_dict[label]
        else:
            raise ValueError(f"Label '{label}' not found in data dictionary.")

    return labeled_array


def create_data_dict(labeled_array, labels):
    """
    Create a dictionary of arrays from a labeled array and a list of labels.

    Args:
        labeled_array (numpy.ndarray): Array of shape (n, m) where m is the number of labels.
        labels (list): List of strings representing the labels.

    Returns:
        dict: Dictionary where keys are labels and values are arrays of length n.
    """
    if labeled_array.shape[1] != len(labels):
        raise ValueError("Number of columns in labeled_array must match the number of labels.")

    data_dict = {}
    for i, label in enumerate(labels):
        data_dict[label] = labeled_array[:, i]

    return data_dict
