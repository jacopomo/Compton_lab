### WORK IN PROGRESS!!!!!!!!

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.interpolate as interp
import time

######## Constanti ########
### Geometria:
L = 11 #Lunghezza collimatore [cm]
DSP = L/2 + 1.5 # Distanza sorgente - plastico [cm]
DBC = 47 # Distanza bersaglio - cristallo [cm]
RC = 2.9 # Raggio del cristallo [cm]
SAAC = np.arctan(RC/DBC) # Semi-apertura angolare del cristallo [rad]
THETA_C = 20 # Angolo al quale si trova il cristallo [gradi]
THETA_C = THETA_C*np.pi/180 # Convertilo il radianti [rad]

### Fisica
ALPHA = 1/137 # Costante di struttura fine
ME = 511 # Massa dell'elettrone [keV]
RE = 1/(ALPHA*ME) # Raggio classico elettrone
E1, E2 = 1173.240, 1332.508 # Energie dei fotoni

### Config
THETA_MIN, THETA_MAX = -np.pi, np.pi # Theta min e max
THETA_MESH = np.linspace(THETA_MIN, THETA_MAX, 10000) # Theta mesh
N_MC = int(1e7) # Num samples
np.random.seed(42) # Seed

######## Classi ########


######## Funzioni ########

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

def campiona_kn(theta_mesh, E, N):
    """"" Campionamento della K-N usando tecniche numeriche

    Parametri:
    theta_mesh: Un mesh di theta da esplorare (linspace theta min-max)
    E: Energia del fotone incidente
    N: Numero di campionamenti
    """""

    kn_norm = kn(theta_mesh, E) / integrate.quad(kn, -np.pi, np.pi, args=(E))[0] #Klein Nishima normalizzata 0-1
    cdf = np.cumsum(kn_norm) * (theta_mesh[1] - theta_mesh[0])  
    cdf = cdf / cdf[-1] 
    inv_cdf = interp.CubicSpline(cdf, theta_mesh)
    u = np.random.uniform(0,1, N)
    x = inv_cdf(u)
    return x

def scatter(E):
    theta_in = np.random.uniform(-4.75, 4.75, N_MC)*np.pi/180
    r = np.arctan(theta_in ) * DSP
    theta_out = theta_in + campiona_kn(THETA_MESH, E, 1)
    pass

######## Monte-Carlo ########
start = time.time()
#### Distribuzione angolare uniforme
theta_in = np.random.uniform(-4.75, 4.75, N_MC)*np.pi/180 # radianti

# Posizione del fotone sul PMT1
r = np.arctan(theta_in ) * DSP

# Aggiorna l'angolo 
theta_out = theta_in + campiona_kn(THETA_MESH, E1, N_MC)

#plt.hist(theta_out, bins=60, color="red", label="theta out")
#plt.legend()
#plt.show()

thetas_accepted=[]
for i in theta_out:
    if (i>THETA_C-SAAC) and (i<THETA_C + SAAC):
        thetas_accepted.append(i)
thetas_accepted = np.array(thetas_accepted)*180/np.pi

print(f'\nAbbiamo utilizzato solo {round(100 * len(thetas_accepted)/N_MC,2)}% dei dati')
print(f'Angoli che vediamo: [{round(min(thetas_accepted),2)}, {round(max(thetas_accepted),2)}] gradi, con il cristallo centrato a {THETA_C*180/np.pi} gradi')
plt.hist(thetas_accepted)



# Timing
end = time.time()
print(f'\nTempo impiegato: {round(end - start,2)}s')
plt.show()