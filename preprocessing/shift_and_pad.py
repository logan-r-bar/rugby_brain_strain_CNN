"""
Shifts and pads time series data to center the peak resultant value.

Original MATLAB implementation can be found here: https://github.com/Jilab-biomechanics/CNN-brain-strains
"""

import numpy as np
from resultant_val import resultant_val

def shift_and_pad(profile, target_idx, cnn_length):
    """
    Shifts the time series data so that the peak resultant value is at the
    target index, and pads the time series to a fixed length.
    
    Args:
        profile (np.ndarray): NxC array of time series data.
        target_idx (int): Target index to center the peak resultant value.
        cnn_length (int): Desired length of the output time series.
        
    Returns:
        padded (np.ndarray): Padded time series of shape (cnn_length, C)."""
    N, C = profile.shape
    res = resultant_val(profile)
    peak_idx = np.argmax(res)
    shift = target_idx - peak_idx
    padded = np.zeros((cnn_length, C))
    start = max(shift, 0)
    end = min(start + N, cnn_length)
    profile_end = end - start
    padded[start:end] = profile[:profile_end]
    if start > 0:
        padded[:start] = profile[0]
    if end < cnn_length:
        padded[end-1:] = profile[-1]
    return padded