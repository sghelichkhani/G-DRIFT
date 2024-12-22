import pytest
import numpy as np
from gdrift.profile import SplineProfile


def test_spline_profile_linear():
    """
    Test SplineProfile for linear interpolation behavior.

    1. Randomly selects two points (x0, y0) and (x1, y1).
    2. Checks if `at_depth` returns correct values for these points.
    3. Validates the interpolated value for a mid-point between x0 and x1.
    """

    # Generate two random points
    x0, x1 = np.random.uniform(0, 100, 2)
    y0, y1 = np.random.uniform(0, 100, 2)

    # Ensure x0 < x1 for monotonicity
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    # Create a SplineProfile
    profile = SplineProfile(depth=np.array([x0, x1]), value=np.array([y0, y1]), spline_type="linear")

    # Check values at x0 and x1
    assert np.isclose(profile.at_depth(x0), y0), f"Value at x0 ({x0}) should be {y0}, got {profile.at_depth(x0)}"
    assert np.isclose(profile.at_depth(x1), y1), f"Value at x1 ({x1}) should be {y1}, got {profile.at_depth(x1)}"

    # Check linear interpolation at the midpoint
    mid_x = (x0 + x1) / 2
    expected_mid_y = (y0 + y1) / 2  # Linear interpolation at the midpoint
    assert np.isclose(profile.at_depth(mid_x), expected_mid_y), (
        f"Interpolated value at midpoint ({mid_x}) should be {expected_mid_y}, got {profile.at_depth(mid_x)}"
    )


def test_spline_profile_extrapolation_enabled():
    """
    Test that SplineProfile correctly extrapolates values when extrapolate=True.
    """
    # Define a simple linear profile
    depths = np.array([0, 100, 200])
    values = np.array([10, 20, 30])

    # Initialize the SplineProfile with extrapolation enabled
    profile = SplineProfile(depth=depths, value=values, extrapolate=True, spline_type="linear")

    # Query a depth outside the range
    depth_outside_range = 300  # Beyond the maximum depth of 200
    expected_value = 40  # Linear extrapolation: slope = (30-20) / (200-100) = 0.1

    # Assert the extrapolated value is as expected
    extrapolated_value = profile.at_depth(depth_outside_range)
    assert np.isclose(extrapolated_value, expected_value), (
        f"Extrapolated value at depth {depth_outside_range} should be {expected_value}, "
        f"but got {extrapolated_value}."
    )


def test_spline_profile_extrapolation_disabled():
    """
    Test that SplineProfile raises an error when extrapolate=False and querying out-of-range values.
    """
    # Define a simple linear profile
    depths = np.array([0, 100, 200])
    values = np.array([10, 20, 30])

    # Initialize the SplineProfile with extrapolation disabled
    profile = SplineProfile(depth=depths, value=values, extrapolate=False, spline_type="linear")

    # Query a depth outside the range
    depth_outside_range = 300  # Beyond the maximum depth of 200

    # Assert that querying out-of-range raises a ValueError
    with pytest.raises(ValueError, match="Depth .* is out of the valid range"):
        profile.at_depth(depth_outside_range)
