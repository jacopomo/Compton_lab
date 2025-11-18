### WORK IN PROGRESS!!!!!!!!

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.interpolate as interp


### Constants
# Lunghezza collimatore
L = 11 #cm

# Distanza sorgente - plastico
DSP = L/2 + 1.5 #cm

# Costante di struttura fine
ALPHA = 1/137

# Massa elettrone
ME = 511 #keV

# Raggio classico elettrone
RE = 1/(ALPHA*ME)





def lambdaratio(theta, E):
    """""
    2Calcola il rapporto lambda su lambda primo
    usato nel calcolo dell'angolo di scattering

    Parametri:
    theta: angolo di scattering
    E: energia del fotone incidente (keV)
   
    """""

    epsilon = E/ME
    return 1/(1+(epsilon*(1-np.cos(theta))))


def kn(theta, E):
    """"" Klein Nishima formula

    Parametri:
    theta: angolo di scattering
    E: energia del fotone incidente (keV)

    Retruns:
    Sezione d'urto differenziale
    """""

    lr=lambdaratio(theta,E)

    return 0.5*(RE**2)*(lr)**2*(lr + (lr)**-1 - (np.sin(theta))**2)

def campiona_kn(theta, E, N):
    """"" Campionamento della K-N usando tecniche numeriche
    """""
    kn_norm = kn(theta, E) / integrate.quad(kn, -np.pi, np.pi, args=(E))[0] #Klein Nishima normalizzata 0-1
    cdf = np.cumsum(kn_norm) * (theta[1] - theta[0])  
    cdf = cdf / cdf[-1] 
    cdf=interp.CubicSpline(theta, cdf)
    u = np.random.uniform(0,1, N)

    return cdf

#### Distribuzione angolare uniforme
theta_in = np.random.uniform(-4.75, 4.75, 10000)*np.pi/180
theta_random = np.random.uniform(-180, 180, 10000)*np.pi/180
theta_mesh = np.linspace(-np.pi, np.pi, 10000)

# Posizione del fotone sul PMT1
r = np.arctan(theta_in ) * DSP


plt.scatter(theta_mesh, campiona_kn(theta_mesh, 1170,1))
plt.show()


