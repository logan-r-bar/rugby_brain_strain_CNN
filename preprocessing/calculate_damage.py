import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

def compute_damage_from_csv(csv_path):
    """
    Compute DAMAGE from a CSV containing:
        time [s]
        acc_x, acc_y, acc_z [rad/s^2]

    Returns:
        DAMAGE (float), delta_norm (np.ndarray), time vector t
    """
    df = pd.read_csv(csv_path)
    t = df.iloc[:, 0].astype(float).to_numpy()

    acc = np.vstack([
        df['ang_x'].to_numpy(),
        df['ang_y'].to_numpy(),
        df['ang_z'].to_numpy()
    ])

    M = np.diag([1.0, 1.0, 1.0])
    kxx, kyy, kzz = 32142.0, 23493.0, 16935.0
    kxy, kyz, kxz = 0.0, 0.0, 1636.3

    K = np.array([
        [kxx + kxy + kxz, -kxy,             -kxz],
        [-kxy,            kxy + kyy + kyz,  -kyz],
        [-kxz,            -kyz,             kxz + kyz + kzz]
    ])

    a1 = 5.9148e-3
    C = a1 * K
    
    # Scale factor
    beta = 2.9903
    
    # Build state-space system
    Minv = np.linalg.inv(M)
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = -Minv @ K
    A[3:6, 3:6] = -Minv @ C

    # Forcing
    def rhs(ti, xi):
        # linear interpolation for acc at ti
        alph = np.vstack([np.interp(ti, t, acc[i]) for i in range(3)])

        delta = xi[0:3]
        delta_dot = xi[3:6]

        xdot = np.zeros(6)
        xdot[0:3] = delta_dot
        xdot[3:6] = -Minv @ (C @ delta_dot + K @ delta) + alph.flatten()
        return xdot

    x0 = np.zeros(6)
    sol = solve_ivp(
        rhs, (t[0], t[-1]), x0,
        t_eval=t,
        method='RK45'   # change to 'Radau' if stiffness warnings appear
    )

    delta = sol.y[0:3, :]  # shape (3, N)
    delta_norm = np.linalg.norm(delta, axis=0)

    damage = beta * np.max(delta_norm)

    return damage


