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

    # RepPad implementation
    if start > 0:
        # Calculate maximum padding count
        max_pad_num = start // (N + 1)
        if max_pad_num > 0:
            # Randomly select a padding count
            pad_num = np.random.randint(1, max_pad_num + 1)
            # Create the repeated sequence
            rep_seq = np.concatenate([profile, np.zeros((1, C))])
            rep_seq = np.tile(rep_seq, (pad_num, 1))
            # Fill the padding area
            pad_len = min(start, len(rep_seq))
            padded[start - pad_len : start] = rep_seq[-pad_len:]

    if end < cnn_length:
        # Calculate maximum padding count
        remaining_len = cnn_length - end
        max_pad_num = remaining_len // (N + 1)
        if max_pad_num > 0:
            # Randomly select a padding count
            pad_num = np.random.randint(1, max_pad_num + 1)
            # Create the repeated sequence
            rep_seq = np.concatenate([np.zeros((1, C)), profile])
            rep_seq = np.tile(rep_seq, (pad_num, 1))
            # Fill the padding area
            pad_len = min(remaining_len, len(rep_seq))
            padded[end : end + pad_len] = rep_seq[:pad_len]

    return padded
