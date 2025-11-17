import os
import pandas as pd
import numpy as np
import itertools
import h5py
from shift_and_pad import shift_and_pad
from conjugate import conjugate_vrot_transform


def process_file(filepath, output_h5_path):
    df = pd.read_csv(filepath)
    profile = df.iloc[:, [4,5,6]].to_numpy()
    cnn_length = 550
    axes_permutations = list(itertools.permutations([0,1,2]))
    target_idx = cnn_length // 2
    base_name = os.path.basename(filepath)
    group_name, _ = os.path.splitext(base_name)

    with h5py.File(output_h5_path, 'a') as hf:
        if group_name in hf:
            print(f"Group '{group_name}' already exists. Overwriting.")
            del hf[group_name]
        group = hf.create_group(group_name)
        
        print(f"Processing {filepath} -> Group '{group_name}' in {output_h5_path}")

        for i, perm in enumerate(axes_permutations):
            permuted = profile[:, perm]
            conj_profile = conjugate_vrot_transform(permuted)
            padded_profile = shift_and_pad(conj_profile, target_idx, cnn_length)
            cnn_input = padded_profile.T[np.newaxis, :, :]
            dataset_name = f"perm_{i+1}"
            group.create_dataset(dataset_name, data=cnn_input)
            print(f"  - Saved dataset '{dataset_name}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process impact data for CNN input")
    parser.add_argument("filepath", type=str)
    parser.add_argument("--output_h5", type=str, default='data/impact_data_augmented.h5')

    args = parser.parse_args()
    process_file(args.filepath, args.output_h5)
