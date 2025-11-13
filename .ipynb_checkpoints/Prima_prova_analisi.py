import numpy as np
import matplotlib.pyplot as plt

file = r"Dati/Prima_misura_prova.dat"

dat = np.loadtxt(file, dtype=int, unpack=True)
xx = np.linspace(0,9000,len(dat))

new = np.array([])
for i,mis in enumerate(dat):
    if mis != 0:
        new = np.concatenate((new, np.full(mis, i)))

plt.hist(new, bins=100)
plt.show()