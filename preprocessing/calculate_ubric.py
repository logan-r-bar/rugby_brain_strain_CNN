"""
Module to compute the UBrIC score from angular acceleration and velocity time series data.
This is a recreation of the Development of a Metric for Predicting Brain Strain Responses Using Head Kinematics paper
Comments should describe the steps taken to compute UBrIC as per the paper.
Paper can be found here: https://doi.org/10.1007/s10439-018-2015-9"""

import numpy as np
import pandas as pd
import math
from scipy.integrate import cumulative_trapezoid


# Critical values for UBrIC calculation (from Table 4 in the reference paper)
w_cr_MPS = np.array([179, 208, 112])          
a_cr_MPS = np.array([13.7e3, 10.1e3, 8.54e3])
r = 2.0 # reccommended value from the paper         

def ubric_term(wp, ap):
    """
    Compute UBrIC term for a single axis.
    This is the part of equation 2 in the paper that is being summed within the square brackets.
    
    Args:
        wp (float): Normalized peak relative velocity.
        ap (float): Normalized peak acceleration.
    Returns:
        float: UBrIC term for the axis.
    """
    ratio = ap / wp
    return wp + (ap - wp) * math.exp(-(ratio))

def acceleration_to_velocity(acc, time):
    """
    Paper requires angular velocity time series, but input data provides angular acceleration.
    This function integrates angular acceleration to obtain angular velocity using
    cumulative trapezoidal integration.

    Args:
        acc (np.ndarray): Angular acceleration time series.
        time (np.ndarray): Time vector corresponding to the acceleration data.

    Returns:
        vel (np.ndarray): Angular velocity time series.
    """
    vel = cumulative_trapezoid(acc, time, initial=0)
    return vel

def compute_ubric(acc_values, vel_values):
    """
    Compute UBrIC score from angular acceleration and velocity time series.

    Args:
        acc_values (np.ndarray): 3xN array of angular acceleration time series for X, Y, Z axes.
        vel_values (np.ndarray): 3xN array of angular velocity time series for X
    Returns:
        ubric (float): Computed UBrIC score.
    """

    # Calculate peak values for each axis
    a_vals = np.max(np.abs(acc_values), axis=1)
    # Calculate peak-to-peak velocity for each axis. Equation 8 in the paper
    w_vals = np.max(np.abs(vel_values), axis=1)

    # Normalize by critical values
    w_prime_MPS = w_vals / w_cr_MPS
    a_prime_MPS = a_vals / a_cr_MPS



    # Compute UBrIC terms for each axis
    t_x_MPS = ubric_term(w_prime_MPS[0], a_prime_MPS[0])
    t_y_MPS = ubric_term(w_prime_MPS[1], a_prime_MPS[1])
    t_z_MPS = ubric_term(w_prime_MPS[2], a_prime_MPS[2])

    # Compute overall UBrIC score (Equation 2)
    ubric_MPS = (t_x_MPS**r + t_y_MPS**r + t_z_MPS**r)**(1/r)
    ubric_nonzero_MPS = max(ubric_MPS, 0) 

    return ubric_nonzero_MPS

def calculate_ubric_from_profile(profile, time):
    """
    Computes UBrIC score from angular acceleration profile.
    Args:
        profile (np.ndarray): N x 3 array of angular acceleration (x, y, z).
        time (np.ndarray): Time vector.
    Returns:
        ubric_score (float): Computed UBrIC score.
    """
    # Transpose to 3xN as expected by compute_ubric
    acc_values = profile.T 
    
    acc_x = acc_values[0]
    acc_y = acc_values[1]
    acc_z = acc_values[2]
     
    vel_x = acceleration_to_velocity(acc_x, time)
    vel_y = acceleration_to_velocity(acc_y, time)
    vel_z = acceleration_to_velocity(acc_z, time)
    vel_values = np.array([vel_x, vel_y, vel_z])
    
    ubric_score = compute_ubric(acc_values, vel_values)
    return ubric_score

def read_impact(path):
    """
    Reads a CSV file containing time series data for angular acceleration,
    computes angular velocity, and calculates the UBrIC score.
    Args:
        path (str): Path to the CSV file.
    Returns:
        ubric_score (float): Computed UBrIC score.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"Unnamed: 0": "time"})
    time_col = "time"
    
    time = df[time_col].astype(float).to_numpy()
    
    # Calculate sampling frequency, assuming uniform sampling
    freq = 1 / (time[1] - time[0])

    acc_x = df["ang_x"].astype(float).to_numpy()
    acc_y = df["ang_y"].astype(float).to_numpy()
    acc_z = df["ang_z"].astype(float).to_numpy()
    
    # Reconstruct profile (N x 3) to use the shared function
    profile = np.column_stack((acc_x, acc_y, acc_z))
    
    return calculate_ubric_from_profile(profile, time)
