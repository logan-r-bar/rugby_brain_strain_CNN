"""
Computes the resultant values from 3D or 4D time series data.

Original MATLAB implementation can be found here: https://github.com/Jilab-biomechanics/CNN-brain-strains
"""

import numpy as np


def resultant_val(val):
    """
    Computes the resultant values from 3D or 4D time series data.

    Args:
        val (np.ndarray): Nx3 or Nx4 array of time series data.

    Returns:
        res (np.ndarray): Resultant values as a 1D array (if input is Nx3)
                          or Nx2 array (if input is Nx4)."""
    val = np.asarray(val)

    if val.shape[1] == 3:
        return np.sqrt(val[:, 0] ** 2 + val[:, 1] ** 2 + val[:, 2] ** 2)
    else:
        res = np.zeros((val.shape[0], 2))
        res[:, 0] = val[:, 0]
        res[:, 1] = np.sqrt(val[:, 1] ** 2 + val[:, 2] ** 2 + val[:, 3] ** 2)
        return res
