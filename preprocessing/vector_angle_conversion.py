"""
Functions to convert between 3D vectors and their corresponding
azimuth and elevation angles. 

Original MATLAB implementation can be found here: https://github.com/Jilab-biomechanics/CNN-brain-strains
"""

import numpy as np


def vec2ang(v):
    """
    Converts a 3D vector to azimuth (theta) and elevation (alpha) angles in degrees.

    Args:
        v (np.ndarray): 3D vector.

    Returns:
        theta (float): Azimuth angle in degrees.
        alpha (float): Elevation angle in degrees.
    """
    theta = np.degrees(np.arctan2(v[1], v[0]))
    alpha = np.degrees(np.arctan2(v[2], np.sqrt(v[0] ** 2 + v[1] ** 2)))
    return theta, alpha


def ang2vec(theta, alpha):
    """
    Converts azimuth (theta) and elevation (alpha) angles in degrees to a 3D vector.

    Args:
        theta (float): Azimuth angle in degrees.
        alpha (float): Elevation angle in degrees.

    Returns:
        np.ndarray: 3D vector.
    """
    a = np.cos(np.radians(alpha)) * np.cos(np.radians(theta))
    b = np.cos(np.radians(alpha)) * np.sin(np.radians(theta))
    c = np.sin(np.radians(alpha))
    return np.array([a, b, c])
