import numpy as np
import json
import re
import pandas as pd
import matplotlib.pyplot as plt

def pq_to_cartesian(p, q):
    """
    Follows the convention in https://www.nature.com/articles/s41586-025-08746-0/figures/3.
    This produces a map of the coordinates that looks consistent with their Fig. 3a.
    """
    h = (q - p) * np.sqrt(3) / 2. # anterior-posterior (higher is more anterior)
    v = (q + p) / 2  # ventro-dorsal (higher is more dorsal)
    return -h, v

def plot_pq(coors, s, label, kwargs):
    # assuming coors is N by 3, where the last column is hemisphere
    x, y = pq_to_cartesian(coors[:, 0].astype(int), coors[:, 1].astype(int))
    plt.scatter(x, y, s=s, label=label, **kwargs)

def format_coordinate_plots():
    plt.xlabel('-h')
    plt.ylabel('v')
    plt.legend(fontsize=8)
    plt.gca().set_aspect('equal', adjustable='box')



def expand_col_entries(df, json_column_name='w.roiInfo', area='ME'):
    """
    Extracts (p, q, post, hemisphere) from a column of JSON-formatted strings
    containing keys like 'ME_R_col_p_q' or 'ME_L_col_p_q', and returns an
    exploded DataFrame with one row per match, preserving original columns.

    Each row in the df is expected to be the summary of weights between a pair
    of neurons. Sometimes these weights exist in multiple columns. In that case,
    in the returned DataFrame, there will be multiple rows for that pair of neurons.

    Parameters:
    - df: pandas DataFrame with a column of JSON strings
    - json_column_name: name of the column containing the JSON strings

    Returns:
    - A new DataFrame with original columns plus 'hemi, h, v' extracted from the JSON data
    """
    
    pattern = re.compile(rf"{re.escape(area)}_([RL])_col_(\d+)_+(\d+)")
    hemi_map = {'L': 0, 'R': 1}
    records = []

    for base in df.to_dict('records'):
        data_string = base.get(json_column_name, '')
        data_dict = json.loads(data_string)

        for key, _val in data_dict.items():
            match = pattern.search(key)
            if not match:
                # in w.roiInfo there are lots of keys not containing coordinate info
                continue
            hemi, hex1_id, hex2_id = match.groups()
            h, v = pq_to_cartesian(int(hex1_id), int(hex2_id))
            records.append({
                **base,
                'hemisphere': hemi,
                'hemi, h, v': (hemi_map[hemi], h, v),
                'column weight': _val['post']
            })

    return pd.DataFrame.from_records(
        records,
        columns=list(df.columns) + ['hemisphere', 'hemi, h, v', 'column weight']
    )