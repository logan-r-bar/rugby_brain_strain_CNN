import numpy as np
from resultant_val import resultant_val
from vector_angle_conversion import vec2ang, ang2vec

def conjugate_rotational_axis(theta, alpha):
    if theta >= 0:
        theta_new = 180 - theta
    else:
        theta_new = -180 - theta
    alpha_new = -alpha
    return theta_new, alpha_new

def conjugate_vrot_transform(profile):
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