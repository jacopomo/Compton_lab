import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.interpolate as interp
import time
import os

######## Constanti ########
### Geometria:
RCOL = 1 # Raggio collimatore [m]
L = 11 #Lunghezza collimatore [cm]
SAACOLL = np.degrees(np.arctan(2/L)) # Semi-apertura angolare del collimatore [gradi]
RP = 2.5 # Raggio plastico [cm]
DSP = 1.5 # Distanza sorgente - plastico [cm]
DBC = 47 # Distanza bersaglio - cristallo [cm]
RC = 2.9 # Raggio del cristallo [cm]
SAAC = np.arctan(RC/DBC) # Semi-apertura angolare del cristallo [rad]
FLUSSO = 2258 # Fotoni al secondo

PHI = 20  # Angolo al quale si trova il cristallo [gradi]


### Fisica
ALPHA = 1/137 # Costante di struttura fine
ME = 511 # Massa dell'elettrone [keV]
RE = 1/(ALPHA*ME) # Raggio classico elettrone
E1, E2 = 1173.240, 1332.508 # Energie dei fotoni

### Config
THETA_MIN, THETA_MAX = -np.pi, np.pi # Theta min e max
THETA_MESH = np.linspace(THETA_MIN, THETA_MAX, 10000) # Theta mesh
STAT_DES = 10000 # Statistica desiderata per l'esperimento
np.random.seed(42) # Seed

N_MC = int(5e7) # Num samples (MASSIMO 5e7 SE NON VUOI FAR DIVENTARE IL TUO COMPUTER UN TERMOSIFONE)

######## Classi ########

class Superficie:
    def __init__(self, raggio, centro=(0,0,0), angolo=0): # Passa gradi
        self.centro = np.array(centro)
        self.angolo = np.radians(angolo)
        self.raggio = raggio

    def normal(self):
        if np.linalg.norm(self.centro)==0:
            normal = np.array([0,0,1])
        else:
            normal = self.centro / np.linalg.norm(self.centro)
        return normal
    
    def pos_sul_piano_unif(self, n, debug_graph=False):
        thetas = np.random.uniform(0,2*np.pi, n)
        rs = self.raggio * np.sqrt(np.random.uniform(0,1,n))
        fx = self.centro[0] + (rs*np.sin(thetas))
        fy = self.centro[1] + (rs*np.cos(thetas)*np.cos(self.angolo))
        fz = self.centro[2] + (rs*np.cos(thetas)*np.sin(self.angolo)) 
        if debug_graph:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(fx,fy,fz) 
            plt.show()
        return fx, fy, fz


class Fotone:
    def __init__(self, energia, px, py, pz, phi, psi): #Passa gradi phi psi
        self.energia = energia
        self.px = px
        self.py = py
        self.pz = pz
        self.phi = np.radians(phi)
        self.psi = np.radians(psi)

    def calcola_int(self, superficie, debug_graph=False, scatter_compton=False):
        p = np.stack((self.px, self.py, self.pz), axis=-1)
        phi, psi = self.phi, self.psi
        centro = superficie.centro
        normal = superficie.normal()

        if scatter_compton:
            scatter_angle = campiona_kn(THETA_MESH, self.energia, len(self.px))
            delta = np.random.uniform(-np.pi/2,np.pi/2, len(self.px)) 
            phi, psi = phi + (scatter_angle*np.cos(delta)), psi + (scatter_angle*np.sin(delta))

        dx, dy, dz = np.sin(psi), np.sin(phi)*np.cos(psi), np.cos(phi)*np.cos(psi)
        d = np.stack((dx, dy, dz), axis=-1)
        num, denom = (centro-p) @ normal, np.sum(d * normal, axis=-1)

        parallel = np.isclose(denom, 0.0, atol=1e-8)
        t = np.full_like(denom, np.nan, dtype=float)

        valid = ~parallel
        t[valid] = num[valid] / denom[valid]
        
        forward = (~np.isnan(t)) & (t >= -1e-8)
        if not np.any(forward):
            return None, None, None, None, None
    
        pts = p + np.expand_dims(t, axis=-1) * d
        pts = pts[forward]
        phi = phi[forward]
        psi = psi[forward]
        if scatter_compton:
            scatter_angle = scatter_angle[forward]

        d2 = np.sum((pts - centro)**2, axis=1)
        mask = d2 < superficie.raggio**2
        if len(pts[mask])==0:
            return None
        xs, ys, zs = pts[mask].T
        phi, psi = phi[mask].T, psi[mask].T
        if scatter_compton:
            scatter_angle = scatter_angle[mask].T

        if debug_graph:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(xs,ys,zs) 
            plt.show()
        if not scatter_compton:
            scatter_angle = np.full(len(xs),None)
        return xs,ys,zs, np.degrees(phi), np.degrees(psi), scatter_angle

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

def mc(E, phi_cristallo=PHI):
    sorgente    = Superficie(RCOL,(0,0,-DSP-L),0)
    collimatore = Superficie(RCOL,(0,0,-DSP), 0)
    plastico    = Superficie(RP, (0,0,0), 0)
    cristallo   = Superficie(RC,(0,DBC*np.sin(np.radians(phi_cristallo)), DBC*np.cos(np.radians(phi_cristallo))), phi_cristallo)

    ## Sorgente - collimatore
    xs, ys, zs = sorgente.pos_sul_piano_unif(N_MC, debug_graph=False) # Genera N punti uniformi sulla sorgente
    phiphi, psipsi = np.random.uniform(-SAACOLL, SAACOLL, len(xs)), np.random.uniform(-SAACOLL, SAACOLL, len(xs)) # Genera angoli uniformi
    f = Fotone(E, xs, ys, zs, phiphi, psipsi) # Genera fotoni 
    xc, yc, zc, phis, psis, _ = f.calcola_int(collimatore, debug_graph=False) # Trova intersezione con collimatore
    #print(f"{round(100*len(xc)/len(xs),2)}% dei fotoni generati escono dal collimatore")

    # Collimatore - plastico
    f = Fotone(E, xc, yc, zc, phis, psis) # Fotoni sul collimatore con l'angolo da prima
    xp, yp, zp, phip, psip, _ = f.calcola_int(plastico, debug_graph=False) # Trova intersezione con plastico
    #print(f"{round(100*len(xp)/len(xc),2)}% dei fotoni uscenti colpiscono il plastico")

    # Plastico - cristallo
    f = Fotone(E, xp, yp, zp, phip, psip)
    xcr, ycr, zcr, _, _, scatter_angles =  f.calcola_int(cristallo, debug_graph=False, scatter_compton=True)
    #print(f"{round(100*len(xc)/len(xp),2)}% dei fotoni dal plastico colpiscono il cristallo")
    
    #print(f"{round(100*len(xcr)/len(xs),2)}% dei fotoni generati colpiscono il cristallo")
    print(f"{round(100*len(xcr)/len(xc),2)}% dei fotoni che escono dal collimatore colpiscono il cristallo")
    #plt.hist(np.degrees(scatter_angles), bins=np.linspace(-90,90,80), label=E, histtype="step")

    print(f"{round((STAT_DES*len(xp))/(FLUSSO*len(xcr)*3600),2)} ore per avere {STAT_DES} eventi")
    energie = compton(E, scatter_angles, 5)
    #plt.hist(energie, bins=np.linspace(energie.min(), energie.max(), 40), label=E, histtype="step")

    return energie, scatter_angles

def plot_compton(phi_cristallo=PHI, plot_scatter_angles=False, all_peaks=False):
    energie1, scatter_angles1 = mc(E1, phi_cristallo)
    energie2, scatter_angles2 = mc(E2, phi_cristallo)

    if plot_scatter_angles:
        angoli = np.concatenate((np.degrees(scatter_angles1),np.degrees(scatter_angles2)))
        plt.figure(figsize=(12,7), dpi=100)
        plt.hist(angoli, color="black", histtype="step", label=f"angoli di scattering [gradi]")
        plt.axvline(np.mean(angoli), color="red", linestyle="--", label=f"Angolo medio: {np.mean(angoli)} gradi")
        plt.title(f"Distribuzione degli angoli di scattering per il cristallo posto a {phi_cristallo} gradi")
        plt.legend(loc="upper right")
        file_path = os.path.join("Montecarlo\Simulazioni", f"simul_dist_{phi_cristallo}gradi.png")
        plt.savefig(file_path)
        plt.show()
        pass

    sommato=np.concatenate((energie1, energie2))
    b_min = min(energie1)
    b_max = max(energie2)
    binss=np.linspace(b_min, b_max, 40)

    plt.figure(figsize=(12,7), dpi=100)
    if all_peaks:
        plt.hist(energie1, bins=binss, color="red", histtype="step", label=f"Picco del fotone di {round(E1,1)}keV")
        plt.hist(energie2, bins=binss, color="blue", histtype="step", label=f"Picco del fotone di {round(E2,1)}keV")
    plt.hist(sommato, bins=binss, color="black", histtype="step", label=f"Somma")
    plt.axvline(compton(E1, np.mean(scatter_angles1)), color="red",  linestyle="--", label=f"Scattering medio: {round(np.mean(np.degrees(scatter_angles1)),1)} gradi, E={round(compton(E1, np.mean(scatter_angles1))[0],1)}keV")
    plt.axvline(compton(E2, np.mean(scatter_angles2)), color="blue", linestyle="--", label=f"Scattering medio: {round(np.mean(np.degrees(scatter_angles2)),1)} gradi, E={round(compton(E2, np.mean(scatter_angles2))[0],1)}keV")
    plt.title(f"Segnale simulato per il cristallo posto a {phi_cristallo} gradi")
    plt.legend(loc="upper right")

    en1 = np.pad(energie1, (0, len(sommato) - len(energie1)), constant_values=np.nan)
    en2 = np.pad(energie2, (0, len(sommato) - len(energie2)), constant_values=np.nan)
    
    file_path = os.path.join("Montecarlo\Simulazioni", f"simul_picchi_{phi_cristallo}gradi.png")
    plt.savefig(file_path)

    file_path = os.path.join("Montecarlo\Simulazioni", f'simul_dati_{phi_cristallo}gradi.csv')
    np.savetxt(file_path, np.column_stack((en1, en2, sommato)), delimiter=',', header="Picco 1, Picco2, Segnale combinato")
    return sommato

######## Monte-Carlo ########
start = time.time()

plot_compton(phi_cristallo=15, plot_scatter_angles=True, all_peaks=True)

end = time.time()
print(f'Tempo impiegato: {round(end - start,2)}s')
plt.show()

