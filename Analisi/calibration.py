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
config_file = os.path.join(tot_path, "_calibration.txt") 

cfg = np.loadtxt(config_file,
                 dtype=str,
                 unpack=True)

calibration_dir = os.path.join(tot_path, cfg[0])
output_dir = os.path.join(tot_path, cfg[1])
calibration_files = u.searchfiles(calibration_dir, "dat")


#Generate a dict for configuration values for calibration files
config_cal_file = os.path.join(calibration_dir, "_config.txt")
config_cal_dict = u.gen_dict(config_cal_file)

#Assign unbinned data to each element's dict
for element,_dict in config_cal_dict.items():
    path = next((x for x in calibration_files if element in x), None)
    unbinned_data = u.unbin(path)
    _dict["data"] = unbinned_data

####################
## Main functions ##
####################

def gauss_expo(x, mu,sigma,A_gauss,lam,A_expo):
    return u.gauss(x, mu,sigma,A_gauss) + u.expo(x, lam,A_expo)

def fit_calibration(dict, graph_name="last_output.png"):
    data = dict["data"]
    try:
        range = dict["range"]
    except:
        range = [0,8192]
    count, bins = np.histogram(data, 150, range=range)
    center_bins = (bins[1:] - bins[:-1])/2

    plt.figure(figsize=(6,4))
    plt.stairs(count,bins, color='gray')
    
    '''
    if dict["fit_enable"]:

        pinit = [2709.0, 89.89, 480.0, 1.0e-3, 340.0]
        bounds = ([0.0, 0.0, 0.0, 0.0, 0.0],
                  [2713.0, 93.0, 500.0, 5.0e-3, 400.0])

        popt, pcov = curve_fit(gauss_expo,
                               center_bins,
                               count,
                               p0=pinit,
                               bounds=bounds)

        print("Fit completato.\n\npopt = ")
        print(popt)
        xx = np.linspace(range[0],range[1], 1000)
        plt.plot(xx, gauss_expo(xx, *pinit), color='red')
        plt.plot(xx, gauss_expo(xx, *popt), color='red')
    '''
    plt.savefig(os.path.join(output_dir, "Graph/" + graph_name), format="png", dpi=600, bbox_inches='tight')
    plt.close()
'''
for element, _dict in config_cal_dict.items():
    fit_calibration(_dict, element+".png")
'''
fit_calibration(config_cal_dict["Cs137"], "Cs137.png")

## ############## ##




