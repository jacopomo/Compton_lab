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
PHI = 20  # Angolo al quale si trova il cristallo [gradi]
FLUSSO = 2258 # Fotoni al secondo


### Fisica
ALPHA = 1/137 # Costante di struttura fine
ME = 511 # Massa dell'elettrone [keV]
RE = 1/(ALPHA*ME) # Raggio classico elettrone
E1, E2 = 1173.240, 1332.508 # Energie dei fotoni

### Config
THETA_MIN, THETA_MAX = -np.pi, np.pi # Theta min e max
THETA_MESH = np.linspace(THETA_MIN, THETA_MAX, 10000) # Theta mesh
N_MC = int(1e4) # Num samples
STAT_DES = 10000 # Statistica desiderata per l'esperimento
np.random.seed(42) # Seed

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

    Returns: Numpy array di N angoli (in radianti) distribuiti secondo la KN normalizzata
    """""

    kn_norm = kn(theta_mesh, E) / integrate.quad(kn, -np.pi, np.pi, args=(E))[0] #Klein Nishima normalizzata 0-1
    cdf = np.cumsum(kn_norm) * (theta_mesh[1] - theta_mesh[0])  
    cdf = cdf / cdf[-1] 
    inv_cdf = interp.CubicSpline(cdf, theta_mesh)
    u = np.random.uniform(0,1, N)
    x = inv_cdf(u)
    return x

def calcola_angolo_cristallo(phi, dbc, dsp):
    """"" Trova l'angolo tra la "sorgente" e il centro del detector, usato ai fini dell'accettanza

    Parametri:
    phi: angolo in gradi sul goiniometro del cristallo
    dbc: distanza bersaglio-cristallo
    dsp: distanza sorgente-plastico

    Returns: l'angolo in radianti
    """""
    phi = np.radians(phi)
    l = np.sqrt(dsp**2+dbc**2-(2*dsp*dbc*np.cos(np.pi-phi)))
    angolo = np.arccos((dsp-(dbc*np.cos(np.pi-phi)))/l)
    return angolo

def distribuzione_sorgente(n, type=0):
    """"" Funzione che seleziona la distribuzione della sorgente di Co

    Parametri: 
    n: Numero di fotoni da generare
    type: Tipo di sorgente. 0 = uniforme, 1 = BOH

    Returns:
    array di angoli in radianti generati dalla distribuzione
    """""
    if type==0:
        theta_in = np.radians(np.random.uniform(-4.75, 4.75, n)) # radianti
    else:
        print("devo ancora implementare altre cose")
    return theta_in

def scatter(E, n):
    """"" Funzione che calcola l'angolo di scattering Compton

    Parametri: 
    E: Energia del fotone incidente [keV]
    n: Numero di fotoni da scatterare

    Returns:
    theta_scattered: L'angolo di scattering Compton (radianti)
    theta_total: L'angolo calcolato dalla sorgente ai fini di contare se arriva sul cristallo (radianti)
    """""
    theta_in = distribuzione_sorgente(n, type=0) # radianti (uniforme)
    theta_scattered = campiona_kn(THETA_MESH, E, n) # radianti
    theta_total = theta_in + theta_scattered
    return theta_scattered, theta_total

######## Monte-Carlo ########
start = time.time()

theta_scattered, theta_total = scatter(E1, N_MC)

plt.hist(theta_total, bins=80)
plt.axvline(calcola_angolo_cristallo(PHI,DBC,DSP), color="red")
plt.axvline(calcola_angolo_cristallo(PHI,DBC,DSP)-SAAC, color="red", linestyle="--", alpha=0.7)
plt.axvline(calcola_angolo_cristallo(PHI,DBC,DSP)+SAAC, color="red", linestyle="--", alpha=0.7)

plt.show()

thetas_accepted=[]
for i in theta_total:
    if (i>np.radians(PHI)-SAAC) and (i<np.radians(PHI) + SAAC):
        thetas_accepted.append(i)
thetas_accepted = np.degrees(np.array(thetas_accepted))


fotoni_visti = len(thetas_accepted)
print(f'\n ========== RISULTATI ==========')
print(f'Abbiamo visto {round(fotoni_visti,2)} fotoni su {int(N_MC)}, ({round(100 * fotoni_visti/N_MC,2)}%)')
print(f'Per vederne {STAT_DES} servierebbero {round(STAT_DES*N_MC/(FLUSSO*60*fotoni_visti),2)} minuti')
print(f'Angoli che vediamo: [{round(min(thetas_accepted),2)}, {round(max(thetas_accepted),2)}] gradi, con il cristallo centrato a {PHI} gradi')
print(f'Delta theta = {round(max(thetas_accepted)-min(thetas_accepted),2)} gradi')
print(f' ===============================\n')
plt.hist(thetas_accepted)



# Timing
end = time.time()
print(f'Tempo impiegato: {round(end - start,2)}s')
plt.show()