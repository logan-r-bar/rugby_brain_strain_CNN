import numpy as np

def resultant_val(val):
    val = np.asarray(val)

    if val.shape[1] == 3:
        return np.sqrt(val[:, 0]**2 + val[:, 1]**2 + val[:, 2]**2)
    else:  
        res = np.zeros((val.shape[0], 2))
        res[:, 0] = val[:, 0]
        res[:, 1] = np.sqrt(val[:, 1]**2 + val[:, 2]**2 + val[:, 3]**2)
        return res