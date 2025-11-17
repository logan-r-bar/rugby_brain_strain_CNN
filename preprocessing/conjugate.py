"""
Functions to compute the conjugate rotational axis and transform the
rotational velocity profile accordingly. 

Original MATLAB implementation can be found here: https://github.com/Jilab-biomechanics/CNN-brain-strains
"""

import numpy as np
from resultant_val import resultant_val
from vector_angle_conversion import vec2ang, ang2vec

def conjugate_rotational_axis(theta, alpha):
    """
    Function to return the conjugate rotational axis so that it produces a
    mirroring brain response about the mid-sagittal plane. Given theta and
    alpha are azimuth and elevation angles in degrees.

    Args:
        theta (float): Azimuth angle in degrees.
        alpha (float): Elevation angle in degrees.

    Returns:
        theta_new (float): Conjugate azimuth angle in degrees.
        alpha_new (float): Conjugate elevation angle in degrees.
    """
    if theta >= 0:
        theta_new = 180 - theta
    else:
        theta_new = -180 - theta
    alpha_new = -alpha
    return theta_new, alpha_new

def conjugate_vrot_transform(profile):
    """
    Function transform the rotational profile to the profile with cojugate
    rotational axis.

    Args:
        profile (np.ndarray): Nx3 array of rotational velocity vectors.
    
    Returns:
        np.ndarray: Transformed profile with conjugate rotational axis.
    """
    t_vrot_rotated = profile.copy()
    res_profile = resultant_val(t_vrot_rotated)
    peak_loc = np.argmax(np.abs(res_profile))
    rot_axis = t_vrot_rotated[peak_loc] / np.linalg.norm(t_vrot_rotated[peak_loc])
    theta, alpha = vec2ang(rot_axis)
    rot_axis_conj = rot_axis.copy()
    if theta < -90 or theta > 90:
        t_theta, t_alpha = conjugate_rotational_axis(theta, alpha)
        rot_axis_conj = ang2vec(t_theta, t_alpha)
    sv = rot_axis_conj / rot_axis
    return t_vrot_rotated * sv