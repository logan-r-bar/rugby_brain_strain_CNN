import numpy as np
from resultant_val import resultant_val

def shift_and_pad(profile, target_idx, cnn_length):
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