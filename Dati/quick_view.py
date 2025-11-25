import numpy as np
import matplotlib.pyplot as plt

file_name = input("Nome file -> ")
file = file_name

resize_factor = int(input("Resize factor dell'istogramma -> "))
bins_tot = 8192
num_bins = int(8192/resize_factor)

scale_log = int(input("Scala (lin = 0, log = 1) -> "))


dat = np.loadtxt(file, dtype=int, unpack=True)
bin_indices = np.arange(dat.size, dtype=int)
unbinned = np.repeat(bin_indices, dat)

if scale_log:
    loglabel = "Logaritmica"
else:
    loglabel = "Lineare"

plt.title(f'Spettro Gamma in Scala {loglabel}')
plt.xlabel(f'Canale')
plt.ylabel('Conteggi (Scala {loglabel})')
if scale_log:
    plt.yscale('log')
count = plt.hist(unbinned, bins=num_bins, histtype='step')
plt.show()