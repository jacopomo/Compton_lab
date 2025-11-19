import numpy as np
import matplotlib.pyplot as plt

file_name = input("Nome file -> ")
file = r"Dati/Calibration/" + file_name

resize_factor = int(input("Resize factor dell'istogramma -> "))
bins_tot = 8192
num_bins = int(8192/resize_factor)


dat = np.loadtxt(file, dtype=int, unpack=True)

new = np.array([])
for i,mis in enumerate(dat):
    if mis != 0:
        new = np.concatenate((new, np.full(mis, i)))


count = plt.hist(new, bins=num_bins, histtype='step')
plt.show()