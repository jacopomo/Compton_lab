import numpy as np
import matplotlib.pyplot as plt
import os

# Prendi un .dat file di misure reali
# Converti conteggi .dat in misure di canale
# Prendi una calibrazione (quella del giorno?)
# Calibra canale -> energia
# Rebinna seguendo un binning predifinito (lo stesso che userò nella simulazione)

# Se richesto, normalizza dividendo in maniera che il conteggio massimo è 1

# Plotta istogramma calibrato e rebinnato
# Return i conteggi e errori poissoniani sui conteggi

# Dopo questi conteggi verranno confrontati con quelli della simulazione (normalizzati)



gradi = input("A quanti gradi? -> ")
file = os.path.join("Dati\Measures", f"{gradi}deg.dat")

num_bins = 256
binning = np.linspace(0,2000, num_bins)

dat = np.loadtxt(file, dtype=int, unpack=True)
bin_indices = np.arange(dat.size, dtype=int)
unbinned = np.repeat(bin_indices, dat)

plt.figure(figsize=(12,7), dpi=100)

energie = (unbinned*0.24725)-3.891 ### Calibrazione presa "al volo"

misure = plt.hist(energie, bins=binning, histtype='step', density=True, label="Segnale misurato", color="black")

p1, p2, tot = np.loadtxt(os.path.join("Montecarlo\Simulazioni", f"simul_dati_{gradi}gradi.csv"), delimiter=",").T
simul = plt.hist(tot, bins=binning, histtype='step', density=True, label="Segnale simulato", color="red")

plt.title(f"Analisi dello spettro a {gradi} gradi")
plt.xlabel(f'Energia [keV]')
plt.ylabel('Conteggi')
plt.legend()
file_path = os.path.join("Dati\Measures", f"confronto_{gradi}deg.png")
plt.savefig(file_path)
plt.show()