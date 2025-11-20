import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
#from scipy.optimize import curve_fit
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
output_dir = os.path.join(tot_path, cfg[1])
calibration_files = u.searchfiles(calibration_dir, "dat")


#Generate a dict for configuration values for calibration files
config_cal_file = os.path.join(calibration_dir, "config.txt")
config_cal_dict = u.gen_dict(config_cal_file)

#Assign unbinned data to each element's dict
for element,_dict in config_cal_dict.items():
    path = next((x for x in calibration_files if element in x), None)
    unbinned_data = u.unbin(path)
    _dict["data"] = unbinned_data

#Fitting function
def fit_calibration(dict):
    data = dict["data"]
    range = dict["range"]
    count, bins = np.histogram(data, 330, range=range)

    plt.figure(figsize=(6,4))
    plt.stairs(count,bins)
    plt.savefig(os.path.join(output_dir, "Graph/output.jpg"), format="jpg", dpi=600, bbox_inches='tight')
    plt.close()

fit_calibration(config_cal_dict["Cs137"])




