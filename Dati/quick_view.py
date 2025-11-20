import numpy as np
import matplotlib.pyplot as plt

file_name = input("Nome file -> ")
file = r"Dati/Calibration/" + file_name

resize_factor = int(input("Resize factor dell'istogramma -> "))
bins_tot = 8192
num_bins = int(8192/resize_factor)


dat = np.loadtxt(file, dtype=int, unpack=True)
bin_indices = np.arange(dat.size, dtype=int)
unbinned = np.repeat(bin_indices, dat)


count = plt.hist(unbinned, bins=num_bins, histtype='step')
plt.show()