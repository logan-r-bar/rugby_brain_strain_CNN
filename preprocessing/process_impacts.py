import os
import itertools
import h5py
import numpy as np
import pandas as pd
from conjugate import conjugate_vrot_transform
from shift_and_pad import shift_and_pad
from calculate_ubric import calculate_ubric_from_profile
from calculate_damage import compute_damage_from_profile

IMPACT_LOCATIONS = [
    'Back', 'Back Left', 'Back Neck', 'Back Right', 'Back Top Left', 'Back Top Right',
    'Bottom Back', 'Bottom Back Left', 'Bottom Back Right', 'Bottom Front', 'Bottom Left',
    'Bottom Right', 'Front', 'Front Bottom Left', 'Front Bottom Right', 'Front Left',
    'Front Neck', 'Front Right', 'Front Top Left', 'Front Top Right', 'Left',
    'Left Neck', 'Right', 'Right Neck', 'Top Back', 'Top Front', 'Top Left',
    'Top Right', 'Unknown'
]

# ---- Precompute constants ----
LOC_TO_IDX = {loc: i for i, loc in enumerate(IMPACT_LOCATIONS)}
UNKNOWN_IDX = LOC_TO_IDX.get('Unknown', None)

CNN_LENGTH = 1000
TARGET_IDX = CNN_LENGTH // 2
AXES_PERMS = ((0, 1, 2),) + tuple(
    p for p in itertools.permutations((0, 1, 2)) if p != (0, 1, 2)
)
AXES_LABELS = ("x", "y", "z")

def one_hot_encode(location):
    enc = np.zeros(len(IMPACT_LOCATIONS), dtype=np.int8)
    if pd.isna(location):
        if UNKNOWN_IDX is not None:
            enc[UNKNOWN_IDX] = 1
        return enc

    idx = LOC_TO_IDX.get(str(location).strip(), UNKNOWN_IDX)
    if idx is not None:
        enc[idx] = 1
    return enc

def process_file(filepath, hf, pred, passed_qa, impact_location):
    """
    Process a single trajectory CSV and write datasets into an already-open HDF5 handle.
    """
    try:
        # Read only required cols; reduce dtype to float32 for speed
        df = pd.read_csv(
            filepath,
            usecols=[0, 1, 2, 3, 4, 5, 6],
            engine="c",
        )

        # time as float64 is fine; profile float32 for compute/write speed
        time = df.iloc[:, 0].to_numpy(dtype=np.float64, copy=False)
        lin_profile = df.iloc[:, 1:4].to_numpy(dtype=np.float32, copy=False)
        rot_profile = df.iloc[:, 4:7].to_numpy(dtype=np.float32, copy=False)

        base_name = os.path.basename(filepath)
        group_name, _ = os.path.splitext(base_name)

        ubric_score = calculate_ubric_from_profile(rot_profile, time)
        damage_score = compute_damage_from_profile(rot_profile, time)

        encoded_location = one_hot_encode(impact_location)

        if group_name in hf:
            del hf[group_name]
        group = hf.create_group(group_name)

        group.attrs["pred"] = pred
        group.attrs["QA"] = passed_qa
        group.attrs["impact_location"] = encoded_location
        group.attrs["ubric_score"] = float(ubric_score)
        group.attrs["damage_score"] = float(damage_score)

        # Write 6 permutations
        for perm in AXES_PERMS:
            rot_permuted = rot_profile[:, perm]
            if perm == (0, 1, 2):
                rot_augmented = rot_permuted
            else:
                rot_augmented = conjugate_vrot_transform(rot_permuted)
            rot_padded = shift_and_pad(rot_augmented, TARGET_IDX, CNN_LENGTH)
            rot_padded = np.asarray(rot_padded, dtype=np.float32)
            rot_cnn_input = rot_padded.T[np.newaxis, :, :]  # (1, 3, 1000)

            lin_permuted = lin_profile[:, perm]
            lin_padded = shift_and_pad(lin_permuted, TARGET_IDX, CNN_LENGTH)
            lin_padded = np.asarray(lin_padded, dtype=np.float32)
            lin_cnn_input = lin_padded.T[np.newaxis, :, :]  # (1, 3, 1000)
            perm_name = "".join(AXES_LABELS[p] for p in perm)
            group.create_dataset(f"perm_{perm_name}", data=rot_cnn_input)
            group.create_dataset(f"lin_perm_{perm_name}", data=lin_cnn_input)

        return True

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def get_h5_path(team_dir, team_name, session_type):
    if session_type == 'Training':
        return os.path.join(team_dir, f"{team_name}_training.h5")
    elif session_type == 'Game':
        return os.path.join(team_dir, f"{team_name}_game.h5")
    return None

def process_all_data():
    root_data_dir = "data"
    unknown_folders = []
    target_team = "ellesmere_u16"

    for team_entry in os.scandir(root_data_dir):
        if not team_entry.is_dir():
            continue
        team_name = team_entry.name
        if team_name != target_team:
            continue
        if team_name == "metadata" or team_name.startswith("."):
            continue

        team_path = team_entry.path
        print(f"Processing Team: {team_name}")

        team_metadata_frames = []

        for session_entry in os.scandir(team_path):
            if not session_entry.is_dir():
                continue
            session_name = session_entry.name
            if session_name.startswith("."):
                continue

            session_path = session_entry.path

            s_lower = session_name.lower()
            if any(x in s_lower for x in ['_game', '_g', 'playoff', 'tournament']):
                session_type = 'Game'
            elif any(x in s_lower for x in ['_practice', '_t', 'training', 'trials']):
                session_type = 'Training'
            else:
                print(f"Skipping unknown session type: {session_name}")
                unknown_folders.append(os.path.join(team_name, session_name))
                continue

            h5_path = get_h5_path(team_path, team_name, session_type)
            if not h5_path:
                continue

            print(f"  Session: {session_name} ({session_type}) -> {h5_path}")

            # Find metadata CSV (first .csv in session root)
            metadata_file = None
            for f in os.scandir(session_path):
                if f.is_file() and f.name.endswith(".csv"):
                    metadata_file = f.path
                    break

            if not metadata_file:
                print(f"    No metadata CSV found in {session_path}")
                continue

            try:
                metadata_df = pd.read_csv(metadata_file, engine="c")
            except Exception as e:
                print(f"    Error reading metadata {metadata_file}: {e}")
                continue

            trajectories_dir = os.path.join(session_path, "trajectories")
            if not os.path.isdir(trajectories_dir):
                print(f"    No trajectories folder in {session_path}")
                continue

            metadata_df.columns = [c.strip() for c in metadata_df.columns]

            id_col = 'Id' if 'Id' in metadata_df.columns else ('_id' if '_id' in metadata_df.columns else None)
            pred_col = 'Pred' if 'Pred' in metadata_df.columns else ('prediction' if 'prediction' in metadata_df.columns else None)
            qa_col = 'Passed QA' if 'Passed QA' in metadata_df.columns else None
            loc_col = 'Impact Location' if 'Impact Location' in metadata_df.columns else ('impact_location' if 'impact_location' in metadata_df.columns else None)

            id_idx = metadata_df.columns.get_loc(id_col) if id_col else None
            pred_idx = metadata_df.columns.get_loc(pred_col) if pred_col else None
            qa_idx = metadata_df.columns.get_loc(qa_col) if qa_col else None
            loc_idx = metadata_df.columns.get_loc(loc_col) if loc_col else None

            if not id_col:
                print(f"    No ID column found in metadata {metadata_file}")
                continue

            trajectory_ids = {
                os.path.splitext(f.name)[0]
                for f in os.scandir(trajectories_dir)
                if f.is_file() and f.name.endswith(".csv")
            }

            # Open the HDF5 ONCE for this session
            count = 0
            session_rows = []
            with h5py.File(h5_path, "a") as hf:
                # itertuples is much faster than iterrows
                for row in metadata_df.itertuples(index=False):
                    impact_id = row[id_idx] if id_idx is not None else None
                    if pd.isna(impact_id):
                        continue

                    impact_id = str(impact_id).strip()
                    if impact_id not in trajectory_ids:
                        continue
                    trajectory_file = os.path.join(trajectories_dir, f"{impact_id}.csv")

                    pred = row[pred_idx] if pred_idx is not None else np.nan
                    passed_qa = row[qa_idx] if qa_idx is not None else np.nan
                    impact_loc = row[loc_idx] if loc_idx is not None else 'Unknown'

                    if process_file(trajectory_file, hf, pred, passed_qa, impact_loc):
                        count += 1
                        session_rows.append(row)

            print(f"    Processed {count} impacts")
            if session_rows:
                session_df = pd.DataFrame.from_records(session_rows, columns=metadata_df.columns)
                team_metadata_frames.append(session_df)

        if team_metadata_frames:
            team_agg_df = pd.concat(team_metadata_frames, ignore_index=True)
            agg_csv_path = os.path.join(team_path, f"{team_name}_all_impacts.csv")
            team_agg_df.to_csv(agg_csv_path, index=False)
            print(f"  Saved aggregated metadata to {agg_csv_path}")

    if unknown_folders:
        with open(os.path.join(root_data_dir, "unknown_folders.txt"), "w") as f:
            for folder in unknown_folders:
                f.write(f"{folder}\n")
        print(f"Logged {len(unknown_folders)} unknown folders to data/unknown_folders.txt")

if __name__ == "__main__":
    process_all_data()
