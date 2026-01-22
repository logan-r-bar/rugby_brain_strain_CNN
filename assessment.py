import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
warnings.filterwarnings("ignore", category=FutureWarning, message=".*np.object.*")

from keras.models import load_model
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import argparse
import time


CNN_LENGTH = 1000
TARGET_IDX = CNN_LENGTH // 2
IMPACT_LOCATIONS = [
    'Back', 'Back Left', 'Back Neck', 'Back Right', 'Back Top Left', 'Back Top Right',
    'Bottom Back', 'Bottom Back Left', 'Bottom Back Right', 'Bottom Front', 'Bottom Left',
    'Bottom Right', 'Front', 'Front Bottom Left', 'Front Bottom Right', 'Front Left',
    'Front Neck', 'Front Right', 'Front Top Left', 'Front Top Right', 'Left',
    'Left Neck', 'Right', 'Right Neck', 'Top Back', 'Top Front', 'Top Left',
    'Top Right'
]


# noinspection PyUnresolvedReferences
def compute_damage_from_profile(profile, t):
    """
    Compute DAMAGE from arrays:
        t: time vector [s], shape (N,)
        profile: angular acceleration, shape (N, 3)
    """
    t = np.asarray(t, dtype=float)
    acc = np.asarray(profile, dtype=float).T

    m = np.diag([1.0, 1.0, 1.0])
    kxx, kyy, kzz = 32142.0, 23493.0, 16935.0
    kxy, kyz, kxz = 0.0, 0.0, 1636.3

    k = np.array([
        [kxx + kxy + kxz, -kxy,             -kxz],
        [-kxy,            kxy + kyy + kyz,  -kyz],
        [-kxz,            -kyz,             kxz + kyz + kzz]
    ])

    a1 = 5.9148e-3
    c = a1 * k

    # Scale factor
    beta = 2.9903

    # Build state-space system
    minv = np.linalg.inv(m)
    a = np.zeros((6, 6))
    a[0:3, 3:6] = np.eye(3)
    a[3:6, 0:3] = -minv @ k
    a[3:6, 3:6] = -minv @ c

    # Forcing
    def rhs(ti, xi):
        # linear interpolation for acc at ti
        alph = np.vstack([np.interp(ti, t, acc[i]) for i in range(3)])

        delta = xi[0:3]
        delta_dot = xi[3:6]

        xdot = np.zeros(6)
        xdot[0:3] = delta_dot
        xdot[3:6] = -minv @ (c @ delta_dot + k @ delta) + alph.flatten()
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

def compute_damage_from_csv(csv_path):
    """
    Compute DAMAGE from a CSV containing:
        time [s]
        acc_x, acc_y, acc_z [rad/s^2]
    """
    df = pd.read_csv(csv_path)
    t = df.iloc[:, 0].astype(float).to_numpy()

    if {'ang_x', 'ang_y', 'ang_z'}.issubset(df.columns):
        profile = np.column_stack([
            df['ang_x'].to_numpy(),
            df['ang_y'].to_numpy(),
            df['ang_z'].to_numpy()
        ])
    else:
        profile = df.iloc[:, 1:4].to_numpy()

    return compute_damage_from_profile(profile, t)

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
    n, c = profile.shape
    res = resultant_val(profile)
    peak_idx = np.argmax(res)
    shift = target_idx - peak_idx
    padded = np.zeros((cnn_length, c))
    start = max(shift, 0)
    end = min(start + n, cnn_length)
    profile_end = end - start
    padded[start:end] = profile[:profile_end]
    if start > 0:
        padded[:start] = profile[0]
    if end < cnn_length:
        padded[end - 1 :] = profile[-1]
    return padded

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
    
def process_file(filepath):
    df = pd.read_csv(
        filepath,
        usecols=[0, 1, 2, 3, 4, 5, 6],
        engine="c",
    )

    # time as float64 is fine; profile float32 for compute/write speed
    time = df.iloc[:, 0].to_numpy(dtype=np.float64, copy=False)
    lin_profile = df.iloc[:, 1:4].to_numpy(dtype=np.float32, copy=False)
    rot_profile = df.iloc[:, 4:7].to_numpy(dtype=np.float32, copy=False)

    # Compute damage
    damage = compute_damage_from_profile(rot_profile, time)

    rot_padded = shift_and_pad(rot_profile, TARGET_IDX, CNN_LENGTH)
    rot_padded = np.asarray(rot_padded, dtype=np.float32)
    lin_padded = shift_and_pad(lin_profile, TARGET_IDX, CNN_LENGTH)
    lin_padded = np.asarray(lin_padded, dtype=np.float32)

    rot_cnn_input = rot_padded.T[np.newaxis, :, :, np.newaxis]  # (1, 3, 1000, 1)
    lin_cnn_input = lin_padded.T[np.newaxis, :, :, np.newaxis]  # (1, 3, 1000, 1)
    sample = np.concatenate([rot_cnn_input, lin_cnn_input], axis=3)  # (1, 3, 1000, 2)

    return sample, damage

def cnn_predict(qa_model_path, location_model_path, sample):
    qa_model = load_model(qa_model_path, compile=False)
    location_model = load_model(location_model_path)
    qa_pred = qa_model.predict(sample)
    qa_label = (qa_pred >= 0.12).astype(bool)
    location_pred = location_model.predict(sample)

    pred_scores = np.asarray(location_pred)
    if pred_scores.ndim == 1:
        pred_scores = pred_scores[np.newaxis, :]

    top3_idx = np.argsort(pred_scores, axis=1)[:, -3:][:, ::-1]
    decoded_top3 = [
        [(IMPACT_LOCATIONS[i], float(pred_scores[row, i])) for i in top3_idx[row]]
        for row in range(pred_scores.shape[0])
    ]
    decoded_label = [IMPACT_LOCATIONS[i] for i in np.argmax(pred_scores, axis=1)]

    return qa_pred, qa_label, decoded_label, decoded_top3

def main():
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(
        description=(
            "Process trajectory CSV files to compute damage and prepare CNN input."
        )
    )
    parser.add_argument("--impact-file", required=True, help="Impact CSV File")
    parser.add_argument("--qa-model", required=True, help="Path to QA CNN model")
    parser.add_argument("--location-model", required=True, help="Path to Impact Location CNN model")
    args = parser.parse_args()

    sample, damage = process_file(args.impact_file)
    qa_pred, qa_label, decoded_label, decoded_top3 = cnn_predict(
        args.qa_model, args.location_model, sample
    )
    if not bool(qa_label[0]):
        print("Impact not predicted to be valid (QA failed).")
    else:
        print("Passed QA.")
        print(f"Predicted Impact Location: {decoded_label[0]}")
        print("Top 3 Predictions:")
        for loc, score in decoded_top3[0]:
            print(f"  {loc}: {score:.4f}")
        print(f"Computed DAMAGE: {damage:.4f}")
    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
