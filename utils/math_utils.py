# Math Utilities
# Created on 6/26/2025

import numpy as np      # deg2rad(), cos(), sin()

SENSOR_OFFSET_MM = 0.080   # Any offset away from the motor shaft will skew scans. Measure this distance with calipers.

def pol_to_cart(rho: float, phi: float, intensity: float | None = None) -> list[float]:
    '''
    Converts coordinates of a point from polar to cartesian system.

    Args:
        rho (float): The radial distance from the origin.
        phi (float): The angular distance from zero degrees.
    Returns:
        The converted point.
    '''
    x = rho * np.cos(np.deg2rad(phi))
    y = rho * np.sin(np.deg2rad(phi))

    return [x, y, intensity] if intensity is not None else [x, y]


def pol_to_cart_array(points: list[list[float]]) -> list[list[float]]:
    '''
    Converts an array of polar (rho, theta) points to the cartesian (x, y) system.

    Args:
        points (list[list[float]]): A list of polar points to be converted.
    Returns:
        cartesian_points (list[list[float]]): A list of converted cartesian points.
    '''
    
    cartesian_points: list[list[float]] = [pol_to_cart(*point) for point in points]

    return cartesian_points


def sph_to_cart(dist: float, l_a: float, m_a: float, i: float | None = None) -> list[float]:
    """
    
    """
    # print(f"D={dist}, LA={l_a}, MA={m_a}, I={i} --> ", end='')

    l_angle: float = np.deg2rad(l_a)				    # lidar angle phi in radians
    m_angle: float = np.deg2rad(m_a)				    # motor angle theta in radians
    
    x: float = dist*np.cos(l_angle)*np.cos(m_angle)	    # x = r*sin(phi)*cos(theta)
    y: float = dist*np.cos(l_angle)*np.sin(m_angle)	    # y = r*sin(phi)*sin(theta)
    z: float = dist*np.sin(l_angle)					    # z = r*cos(phi)
    
    x += SENSOR_OFFSET_MM*np.sin(m_angle)		# Correct for horizontal lidar offset
    y += SENSOR_OFFSET_MM*np.cos(m_angle)		# Correct for horizontal lidar offset

    point = [round(dim, 4) for dim in ([x, y, z, i] if i is not None else [x, y, z])]

    return point
    

def sph_to_cart_array(points: list[list[float]]) -> list[list[float]]:

    """
    Vectorized conversion of the full LiDAR cloud from spherical
    coordinates to Cartesian coordinates.

    Input:
        [distance, lidar_angle, motor_angle, intensity]

    Output:
        [x, y, z, intensity]
    """

    if not points:
        return []

    arr = np.asarray(
        points,
        dtype=float
    )

    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(
            f"Expected LiDAR points shaped (N, 3+) "
            f"but got {arr.shape}"
        )

    dist = arr[:, 0]

    lidar_angle = np.deg2rad(
        arr[:, 1]
    )

    motor_angle = np.deg2rad(
        arr[:, 2]
    )

    cos_lidar = np.cos(
        lidar_angle
    )

    x = (
        dist
        * cos_lidar
        * np.cos(motor_angle)
    )

    y = (
        dist
        * cos_lidar
        * np.sin(motor_angle)
    )

    z = (
        dist
        * np.sin(lidar_angle)
    )

    # Same LiDAR offset correction used by sph_to_cart()
    x += (
        SENSOR_OFFSET_MM
        * np.sin(motor_angle)
    )

    y += (
        SENSOR_OFFSET_MM
        * np.cos(motor_angle)
    )

    xyz = np.column_stack(
        (x, y, z)
    )

    xyz = np.round(
        xyz,
        4
    )

    # Keep intensity if present
    if arr.shape[1] >= 4:

        intensity = arr[:, 3:4]

        output = np.column_stack(
            (xyz, intensity)
        )

    else:

        output = xyz

    return output.tolist()