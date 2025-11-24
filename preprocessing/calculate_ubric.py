""" Updated UBrIC model with better comprehension of time series data provided by HitIQ

Version: 2.0 - revamped for full time series data and complex UBrIC utilization
Author: Meriem Hcini    
Date: October 2025
"""

# imports 
import os 
import numpy as np
import pandas as pd


# function to read impact csv
def read_impact(path):
    """
    Read impact data from a CSV file.
    Fixes unamed first column by renaming it to 'time'.

    Parameters:
    path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: DataFrame containing the impact data.
    """
    df = pd.read_csv(path)
    df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)  # Rename the first column to 'time'
    features = extract_features(df)
    ubric_score = compute_ubric(features)
    return ubric_score

# simple feature extraction function (resultant peaks)
def extract_features(df):
    """
    Extract features from the impact data DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the impact data.

    Returns:
    dict: Dictionary containing extracted features.
    """
    features = {}
    # Example feature: peak linear acceleration magnitude
    df["lin_mag"] = np.sqrt(df["lin_x"]**2 + df["lin_y"]**2 + df["lin_z"]**2)
    features['peak_lin_mag'] = df["lin_mag"].max()

    # Example feature: peak angular velocity magnitude
    df["ang_mag"] = np.sqrt(df["ang_x"]**2 + df["ang_y"]**2 + df["ang_z"]**2)
    features['peak_ang_mag'] = df["ang_mag"].max()

    return features

# UBrIC computation function
def compute_ubric(features):
    """
    Compute the UBrIC metric from extracted features.

    Parameters:
    features (dict): Dictionary containing extracted features.

    Returns:
    float: Computed UBrIC value.
    """
    # example coefficients - need to fix this computation
    beta0, beta1, beta2 = 0.1, 0.05, 0.03

    # Compute UBrIC
    ubric = (beta0 + 
             beta1 * features['peak_lin_mag'] + 
             beta2 * features['peak_ang_mag'])
    
    return ubric

