from scipy import signal

def filter_and_detrend(data, cutoff=300.0, fs=3200.0, order=4):
    """
    Zero-phase 4th-order Butterworth filter for head kinematics.
    Correctly normalizes cutoff frequency using fs.
    
    Supports both 1D arrays (time series) and 2D arrays (time x channels).
    If 2D, filtering is applied along axis 0 (time).
    """
    data_detrended = signal.detrend(data, axis=0)
    b, a = signal.butter(order, cutoff, btype='low', fs=fs)
    y = signal.filtfilt(b, a, data_detrended, axis=0)
    return y
