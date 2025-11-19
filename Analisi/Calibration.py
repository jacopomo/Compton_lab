import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
config_cal_file = os.path.join(calibration_dir, "config.txt")

#Generate a dict for configuration values for calibration files
config_cal_dict = u.gen_dict(config_cal_file)

for element,_dict in config_cal_dict.items():
    path = next((x for x in calibration_files if element in x), None)
    unbinned_data = u.unbin(path)
    _dict["data"] = unbinned_data

