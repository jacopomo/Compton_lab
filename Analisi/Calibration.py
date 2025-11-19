import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import utils as u

#Setupping directories and files' path
tot_path = os.path.dirname(__file__)
config_file = os.path.join(tot_path, "config.txt") 

cfg = np.loadtxt(config_file,
                 dtype=str,
                 unpack=True)

calibration_dir = os.path.join(tot_path, cfg[0])
calibration_files = u.searchfiles(calibration_dir, "dat")
config_cal_file = os.path.join(calibration_dir, "config.txt")

#Generate a dict for configuration values for calibration files
df = pd.read_csv(config_cal_file,
                 sep=r"\s+",
                 engine="python",
                 dtype=str)
df = df.replace({np.nan: None})

value = {}
key_col = df.columns[0]

for _,row in df.iterrows():
    main_key = row[key_col]
    nested = {}

    for col,val in row.items():
        if col == key_col:
            continue
        if val is not None:
            nested[col] = val
    value[main_key] = nested



