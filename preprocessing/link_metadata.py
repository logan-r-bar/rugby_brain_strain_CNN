import pandas as pd
import re
import os

def get_metadata_prediction(impact_filepath):
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
        
        # Filter by pred value if specified
        if pred_value_to_find is not None:
            metadata_df = metadata_df[metadata_df['pred'] == pred_value_to_find]

        matching_rows = metadata_df[(metadata_df['team_code'] == team_code_str) & (metadata_df['id'] == id_str)]
        if instance < len(matching_rows):
            return matching_rows.iloc[instance]['pred']

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
            return matching_rows.iloc[instance]['pred']
        
        i += 1
    
    raise IndexError(f"Instance {instance} not found for team_code {team_code_str} and id {id_str} in any metadata file for {suffix} with pred={pred_value_to_find}")
