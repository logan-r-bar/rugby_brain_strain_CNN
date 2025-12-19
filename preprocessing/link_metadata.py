import pandas as pd
import re
import os
import numpy as np

# List of all possible impact locations
IMPACT_LOCATIONS = [
    'Back', 'Back Left', 'Back Neck', 'Back Right', 'Back Top Left', 'Back Top Right',
    'Bottom Back', 'Bottom Back Left', 'Bottom Back Right', 'Bottom Front', 'Bottom Left',
    'Bottom Right', 'Front', 'Front Bottom Left', 'Front Bottom Right', 'Front Left',
    'Front Neck', 'Front Right', 'Front Top Left', 'Front Top Right', 'Left',
    'Left Neck', 'Right', 'Right Neck', 'Top Back', 'Top Front', 'Top Left',
    'Top Right', 'Unknown'
]

def get_metadata(impact_filepath):
    """
    Parses an impact file path to find the corresponding metadata entry and return the 'pred' and 'ubric' values.
    It handles cases where metadata is split across multiple files and filters by prediction value based on the directory.

    Args:
        impact_filepath (str): The path to the impact data file.

    Returns:
        tuple: A tuple containing the 'pred' and 'ubric' values from the metadata file.
    """
    base_name = os.path.basename(impact_filepath)
    
    # Correctly determine the directory containing 'pred_true' or 'pred_false'
    path_parts = impact_filepath.split(os.sep)
    pred_value_to_find = None
    if 'pred_true' in path_parts:
        pred_value_to_find = True
    elif 'pred_false' in path_parts:
        pred_value_to_find = False

    # Regex to extract identifiers from the filename
    match = re.match(r'(\d+)_([\w\d]+)_(g\d+|tw\d+)(?:_(\d+))?\.csv', base_name)
    if not match:
        raise ValueError(f"Filename {base_name} does not match expected pattern.")

    team_code_str, id_str, suffix, instance_str = match.groups()
    instance = int(instance_str) if instance_str else 0

    # Try base metadata file first
    metadata_filename = f"metadata_{suffix}.csv"
    metadata_filepath = os.path.join('data', 'metadata', metadata_filename)

    if os.path.exists(metadata_filepath):
        metadata_df = pd.read_csv(metadata_filepath, dtype={'id': str, 'team_code': str})
        metadata_df = metadata_df[metadata_df['pred'] == pred_value_to_find]

        matching_rows = metadata_df[(metadata_df['team_code'] == team_code_str) & (metadata_df['id'] == id_str)]
        if instance < len(matching_rows):
            pred_val = matching_rows.iloc[instance]['pred']
            impact_location = matching_rows.iloc[instance]['impact_location']
            ubric_score = matching_rows.iloc[instance]['ubric']
            
            # One-hot encode the impact location
            encoded_location = one_hot_encode(impact_location, IMPACT_LOCATIONS)
            
            return pred_val, encoded_location, ubric_score

    # If not found, check for numbered suffixes
    i = 1
    while True:
        metadata_filename_numbered = f"metadata_{suffix}_{i}.csv"
        metadata_filepath_numbered = os.path.join('data', 'metadata', metadata_filename_numbered)
        if not os.path.exists(metadata_filepath_numbered):
            # No more numbered files to check
            break

        metadata_df = pd.read_csv(metadata_filepath_numbered, dtype={'id': str, 'team_code': str})

        # Filter by pred value if specified
        if pred_value_to_find is not None:
            metadata_df = metadata_df[metadata_df['pred'] == pred_value_to_find]

        matching_rows = metadata_df[(metadata_df['team_code'] == team_code_str) & (metadata_df['id'] == id_str)]

        if instance < len(matching_rows):
            pred_val = matching_rows.iloc[instance]['pred']
            impact_location = matching_rows.iloc[instance]['impact_location']
            ubric_score = matching_rows.iloc[instance]['ubric']
            # One-hot encode the impact location
            encoded_location = one_hot_encode(impact_location, IMPACT_LOCATIONS)

            return pred_val, encoded_location, ubric_score
        
        i += 1
    
    raise IndexError(f"Instance {instance} not found for team_code {team_code_str} and id {id_str} in any metadata file for {suffix} with pred={pred_value_to_find}")

def one_hot_encode(location, locations_list):
    """
    One-hot encodes a single location string into a vector.
    
    Args:
        location (str): The impact location string to encode.
        locations_list (list): The list of all possible location strings.
        
    Returns:
        np.ndarray: A one-hot encoded vector.
    """
    # Create a zero vector
    encoding = np.zeros(len(locations_list), dtype=int)
    
    # Find the index of the location and set that element to 1
    try:
        index = locations_list.index(location)
        encoding[index] = 1
    except ValueError:
        # Handle the case where the location is not in the list
        # This could be by raising an error, logging a warning, or assigning to an 'unknown' category
        print(f"Warning: Location '{location}' not found in the predefined list.")
        # Optionally, you could have an 'Unknown' category at a fixed index (e.g., the last one)
        # and set that to 1 if the location is not found.
    
    return encoding
