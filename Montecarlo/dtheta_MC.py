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
PHI = 15  # Angolo al quale si trova il cristallo [gradi]
FLUSSO = 2258 # Fotoni al secondo


### Fisica
ALPHA = 1/137 # Costante di struttura fine
ME = 511 # Massa dell'elettrone [keV]
RE = 1/(ALPHA*ME) # Raggio classico elettrone
E1, E2 = 1173.240, 1332.508 # Energie dei fotoni

### Config
THETA_MIN, THETA_MAX = -np.pi, np.pi # Theta min e max
THETA_MESH = np.linspace(THETA_MIN, THETA_MAX, 10000) # Theta mesh
N_MC = int(1e6) # Num samples
STAT_DES = 10000 # Statistica desiderata per l'esperimento
np.random.seed(42) # Seed

######## Classi ########

class Superficie:
    def __init__(self, raggio, posizione=(0,0,0), angolo=0): # Passa gradi
        self.posizione = posizione
        self.angolo = np.radians(angolo)
        self.raggio = raggio

    def genera_uniforme(self, n):
        thetas = np.random.uniform(0,2*np.pi, n)
        rs = self.raggio * np.sqrt(np.random.uniform(0,1,n))
        distribuzione_sorgente(n, type=0)
        pass

class Fotone:
    def __init__(self, energia, posizione=[0,0,0], direzione=[0,0]): #Passa gradi phi psi
        self.energia = energia
        self.posizione = np.array(posizione)
        self.direzione = np.radians(np.array(direzione))
    
    def scatter_uniforme(self):
        pass

    def scatter_compton(self):
        pass
    
    def calcola_int(self, superficie):
        posizione = self.posizione
        direzione = self.direzione
        centro = np.array(superficie.posizione)
        if np.linalg.norm(centro)==0:
            normal = np.array([1,0,0])
        else:
            normal = centro / np.linalg.norm(centro)

        if direzione[0]==0:
            direzione[0]==0.01
        if direzione[1] == 0:
            direzione[1] == 0.01

        y_int =  ((normal[1]*centro[1]) + (normal[2]*(centro[2] -posizione[2] - (posizione[1]/np.tan(direzione[0])))))/((normal[1]) + (normal[2]/np.tan(direzione[0])))
        z_int = ((y_int-posizione[1])/np.tan(direzione[0])) + posizione[2]
        x_int = ((z_int-posizione[2]) * np.tan(direzione[1])) + posizione[0]

        intersezione = np.array([x_int, y_int, z_int])

        if np.sum(((intersezione-centro)**2)) < superficie.raggio:
            print("intersecato")
            return intersezione
        else:
            print("non intersecato")
            return None


######## Funzioni ########

def kn(theta, E):
    """"" Klein Nishima formula

    Parametri:
    theta: angolo di scattering
    E: energia del fotone incidente (keV)

    Retruns:
    Sezione d'urto differenziale
    """""
    epsilon = E/ME
    lr=1/(1+(epsilon*(1-np.cos(theta))))

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
    type: Tipo di sorgente. 0 = uniforme, 1 = gaussiana, 2 = lineare

    Returns:
    array di angoli in radianti generati dalla distribuzione
    """""
    if type==0:
        theta_in = np.radians(np.random.uniform(-4.75, 4.75, n)) # radianti
    elif type==1:
        theta_in = np.radians(np.random.normal(0,4.75/3, n)) # radians
    elif type==2:
        theta_in = np.full(n, 0)
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
    theta_in = distribuzione_sorgente(n, type=0) # radianti (0: uniforme, 1: gauss, 2: ideale)
    theta_scattered = campiona_kn(THETA_MESH, E, n) # radianti
    theta_total = theta_in + theta_scattered
    return theta_scattered, theta_total

def accept(E):
    """"" Funzione che calcola se un raggio viene visto dal cristallo o no

    Parametri:
    theta_scattered: numpy array di angoli di scattering
    theta_total: numpy array di angoli misurati dalla sorgente

    Returns:
    numpy array di angoli di scattering considerando solo quelli visti dal cristallo
    """""
    theta_scattered, theta_total = scatter(E, N_MC)
    thetas_accepted=[]
    counter = 0
    for i in theta_total:
        if (i>np.radians(PHI) - SAAC) and (i<np.radians(PHI) + SAAC):
            thetas_accepted.append(theta_scattered[counter])
        counter += 1
    thetas_accepted = np.degrees(np.array(thetas_accepted))
    return thetas_accepted

def seen_thetas(picco=1):
    """"" Genera il grafico per gli angoli di scattering visti in base al picco

    Parametri:
    picco: 1 o 2 in base al picco che si vuole vedere

    Returns: None
    """""
    if picco == 1:
        thetas_accepted=accept(E1)
    if picco == 2:
        thetas_accepted=accept(E2)
    else:
        print("Picco deve essere 1 o 2")
        end

    fotoni_visti = len(thetas_accepted)
    print(f'\n ========== RISULTATI ==========')
    print(f'Abbiamo visto {round(fotoni_visti,2)} fotoni su {int(N_MC)}, ({round(100 * fotoni_visti/N_MC,2)}%)')
    print(f'Per vederne {STAT_DES} servierebbero {round(STAT_DES*N_MC/(FLUSSO*60*fotoni_visti),2)} minuti')
    print(f'Angoli che vediamo: [{round(min(thetas_accepted),2)}, {round(max(thetas_accepted),2)}] gradi, con il cristallo centrato a {PHI} gradi')
    print(f'Delta theta = {round(max(thetas_accepted)-min(thetas_accepted),2)} gradi')
    print(f' ===============================\n')


    counts=plt.hist(thetas_accepted, bins=30)
    plt.axvline(PHI-np.degrees(SAAC), color="red", linestyle="--", label="Detector aperature")
    plt.axvline(PHI+np.degrees(SAAC), color="red", linestyle="--")
    half_max = max(counts[0]) / 2
    indices = np.where(counts[0] >= half_max)[0]
    fwhm = counts[1][indices[-1]+1] - counts[1][indices[0]]
    plt.axvline(counts[1][indices[0]], color="black", linestyle="--", label="Full width half maximum")
    plt.axvline(counts[1][indices[-1]+1], color="black", linestyle="--")
    plt.axvline(np.mean(thetas_accepted), color="yellow", label=f"Mean angle: {round(np.mean(thetas_accepted),2)}")
    plt.legend()
    plt.show()

def compton(E, theta, errore=0):
    """"" Calcola l'energia di un fotone entrante con energia E ed angolo theta

    Parametri:
    E: Energia in ingresso del fotone
    theta: angolo in ingresso (in radianti) del fotone
    errore: errore gaussiano strumentale (keV)

    Returns:
    Energia del fotone dopo lo scattering
    """""
    return E/(1+(E*(1-np.cos(theta))/ME)) + np.random.normal(0,errore,1)

def plot_compton():
    thetas_accepted1 = accept(E1)
    thetas_accepted2 = accept(E2)

    
    compton1=compton(E1, np.radians(thetas_accepted1),5)
    compton2=compton(E2, np.radians(thetas_accepted2),5)

    sommato=np.concatenate((compton1, compton2))
    b_min = min(compton1)
    b_max = max(compton2)
    binss=np.linspace(b_min, b_max, 50)

    #plt.hist(compton1, bins=binss, color="red", histtype="step", label=f"Picco a {round(E1,2)}keV")
    #plt.hist(compton2, bins=binss, color="blue", histtype="step", label=f"Picco a {round(E2,2)}keV")
    plt.hist(sommato, bins=binss, color="black", histtype="step", label=f"Somma")
    plt.axvline(compton(E1, np.radians(np.mean(thetas_accepted1))), color="red", linestyle="--", label=f"{compton(E1, np.radians(np.mean(thetas_accepted1)))}keV")
    plt.axvline(compton(E2, np.radians(np.mean(thetas_accepted2))), color="blue", linestyle="--", label=f"{compton(E2, np.radians(np.mean(thetas_accepted2)))}keV")
    plt.legend()
    plt.show()
    pass

######## Monte-Carlo ########
start = time.time()

#plot_compton()
sorgente    = Superficie(1,(0,0,-DSP-L),0)
collimatore = Superficie(1,(0,0,-DSP), 0)
plastico    = Superficie(10, (0,0,0), 0)
cristallo   = Superficie(RC,(0,DBC*np.sin(np.radians(PHI)), DBC*np.cos(np.radians(PHI))), PHI)

f=Fotone(E1, [0,0,0], [0,0])
f.calcola_int(cristallo)

end = time.time()
print(f'Tempo impiegato: {round(end - start,2)}s')
plt.show()



