import numpy as np
import matplotlib.pyplot as plt

me = 0.511          #Electron's mass in MeV
EE = [1.17, 1.33]   #Photons energies in MeV
sigma = 0.05        #Resolution in MeV
n = 1_000_000       #Events
f = 0.16            #Fraction of direct absorbation and Compton effect events

E_ph = [np.random.normal(E,sigma,int(n*f)) for E in EE]

def ComptonEnergy(cos, Ein):
    return Ein / (1 + (Ein / me) * (1 - cos))

def cross(cos, Ein):
    Eout = ComptonEnergy(cos, Ein)
    R = Eout / Ein
    return Eout, R**2 * (R + 1/R - 1 + cos**2)


E_cmpt = [[],[]]
E_e = [[],[]]
for i,Ein in enumerate(EE):
    cos_random = np.random.uniform(-1,1,n)
    E_random, p = cross(cos_random, Ein)
    mask = np.random.uniform(0,2,n) < p

    N = len(cos_random[mask])
    E_cmpt[i] = E_random[mask] + np.random.normal(0,sigma,N)
    E_e[i] = np.full(N,Ein) - E_random[mask] + np.random.normal(0,sigma,N)


E_tot = np.concatenate([arr for _list in (E_ph, E_cmpt, E_e) for arr in _list])
E_sgn = np.concatenate([arr for arr in E_ph])
E_bkg = np.concatenate([arr for _list in (E_cmpt, E_e) for arr in _list])

m = 300     #number of bins
r = [0,1.6] #energy's range in MeV

plt.hist(E_ph, m, histtype='step',range=r, label="Direct-absorbed photons")
#plt.hist(E_cmpt, m, histtype='step',range=r, label="Scattered photons")
#plt.hist(E_e, m, histtype='step',range=r, label="Scattered electrons")

plt.hist(E_tot, m, histtype='step',range=r, label="Total signal")
plt.hist(E_bkg, m, histtype='step',range=r, label="Total background")
#plt.hist(E_sgn, m, histtype='step',range=r, label="signal-background")

plt.title("Montecarlo di fotoni incidenti sullo scintillatore inorganico")
plt.xlabel("Energy [MeV]")
plt.ylabel("Counts")

plt.show()