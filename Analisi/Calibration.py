import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import curve_fit
import os
import time

import utils as u

#Setupping directories and files' path
tot_path = os.path.dirname(__file__)
config_file = os.path.join(tot_path, "config.txt") 

cfg = np.loadtxt(config_file,
                 dtype=str,
                 unpack=True)

calibration_dir = os.path.join(tot_path, cfg[0])
calibration_files = u.searchfiles(calibration_dir, "dat")

file_map = {}
for f in calibration_files:
    base = os.path.basename(f)
    stem = os.path.splitext(base)[0]
    file_map[stem] = f


#Generate a dict for configuration values for calibration files
config_cal_file = os.path.join(calibration_dir, "config.txt")
config_cal_dict = u.gen_dict(config_cal_file)

#Assign unbinned data to each element's dict
for element, _dict in config_cal_dict.items():
    path = file_map.get(element)
    if path is None:
        continue
    unbinned_data = u.unbin(path)
    _dict["data"] = unbinned_data




