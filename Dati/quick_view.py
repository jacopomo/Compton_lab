import numpy as np
import matplotlib.pyplot as plt

file_name = input("Nome file -> ")
file = r"Dati/Calibration/Background/" + file_name

resize_factor = int(input("Resize factor dell'istogramma -> "))
bins_tot = 8192
num_bins = int(8192/resize_factor)


dat = np.loadtxt(file, dtype=int, unpack=True)
bin_indices = np.arange(dat.size, dtype=int)
unbinned = np.repeat(bin_indices, dat)

plt.title(f'Spettro Gamma in Scala Logaritmica')
plt.xlabel(f'Canale')
plt.ylabel('Conteggi (Scala Logaritmica)')
plt.yscale('log')
count = plt.hist(unbinned, bins=num_bins, histtype='step')
plt.show()