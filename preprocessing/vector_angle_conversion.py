import numpy as np

def vec2ang(v):
    theta = np.degrees(np.arctan2(v[1], v[0]))
    alpha = np.degrees(np.arctan2(v[2], np.sqrt(v[0]**2 + v[1]**2)))
    return theta, alpha

def ang2vec(theta, alpha):
    a = np.cos(np.radians(alpha)) * np.cos(np.radians(theta))
    b = np.cos(np.radians(alpha)) * np.sin(np.radians(theta))
    c = np.sin(np.radians(alpha))
    return np.array([a, b, c])