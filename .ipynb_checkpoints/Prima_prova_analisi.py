import numpy as np
import matplotlib.pyplot as plt

file = r"hist_Co60.dat"

dat = np.loadtxt(file, dtype=int, unpack=True)

new = np.array([])
for i,mis in enumerate(dat):
    if mis != 0:
        new = np.concatenate((new, np.full(mis, i)))

plt

plt.hist(new, bins=4096)
plt.show()