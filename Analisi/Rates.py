import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import UnivariateSpline
import os
import json

_base = os.getcwd()
_dir = "Dati"
_subdir = "Measures"
_finaldir = "Rates"
_filename = "Counts.json"

config_path = os.path.join(_base, _dir, _subdir, _finaldir, _filename)

with open(config_path, 'r') as f:
        config = json.load(f)

dati = {k: np.array(v) for k, v in config["Dati"].items()}
deg, pmt1, pmt2, both, delta_t = dati.values()
tau = 113 * 1e-6

rate1 = pmt1/delta_t
rate2 = pmt2/delta_t
rate_both = both/delta_t

sigma_rate1 = np.sqrt(pmt1)/delttaua_t
sigma_rate2 = np.sqrt(pmt2)/delta_t
sigma_rateboth = np.sqrt(both)/delta_t

rate_acc = rate1 * rate2 * tau


fig, ax1 = plt.subplots()

ax1.errorbar(deg, rate_both*1e3, yerr=sigma_rateboth*1e3, fmt='.', label='Rate coincidenze')
ax1.errorbar(deg, rate_acc*1e3, fmt='.', label='Rate accidentali')

ax1.set_ylabel('Rate delle coincidenze [#/s]')
ax1.set_xlabel('Angolo [deg]')
ax1.grid()

'''
ax2 = ax1.twinx()
ax2.errorbar(deg, rate1, yerr=sigma_rate1, fmt='.', label='Rate PMT1', color='r')
ax2.errorbar(deg, rate2, yerr=sigma_rate2, fmt='.', label='Rate PMT2', color='black')
ax2.set_ylabel('Rate PMT2 [#/ms]')
ax2.grid(linestyle='--')
'''

fig.legend()
plt.show()