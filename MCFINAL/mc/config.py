import numpy as np

N_MC = int(5e6) # Num samples (MASSIMO 5e7 SE NON VUOI FAR DIVENTARE IL TUO COMPUTER UN TERMOSIFONE)

### Geometria:
RCOL = 1 # Raggio collimatore [cm]
L = 11 #Lunghezza collimatore [cm]
SAACOLL = np.degrees(np.arctan(2/L)) # Semi-apertura angolare del collimatore [gradi]
RP = 2.5 # Raggio plastico [cm]
LP = 3 # Lunghezza plastico [cm]
DSP = 1.5 # Distanza sorgente - plastico [cm]
DBC = 47 # Distanza bersaglio - cristallo [cm]
RC = 2.54 # Raggio del cristallo [cm]
LC = 5.08 # Lunghezza del cristallo [cm]
SAAC = np.arctan(RC/DBC) # Semi-apertura angolare del cristallo [rad]
FLUSSO = 2258 # Fotoni al secondo

PHI = 20  # Angolo al quale si trova il cristallo [gradi]


### Fisica
ALPHA = 1/137 # Costante di struttura fine
ME = 511 # Massa dell'elettrone [keV]
RE = 2.8179403262e-13 #[cm]
E1, E2 = 1173.240, 1332.508 # Energie dei fotoni

### Config
THETA_MIN, THETA_MAX = 0, np.pi # Theta min e max
THETA_MESH = np.linspace(THETA_MIN, THETA_MAX, 250) # Theta mesh
E_ref = np.linspace(1, 2000, 100)  # 100 bins da 1 keV a 2000 keV
E_ref = np.concatenate(([E1, E2], E_ref))  # Energie importanti

ESOGLIA_C = 550 # Soglia del cristallo [keV]
EBINMAX = 2000 # Massimo del binning [keV]
NBINS = 80

MAX_CML_TRIES = 500  # Maximum times to re-sample cml for escaping photons

#np.random.seed(42) # Seed