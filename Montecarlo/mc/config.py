import numpy as np

### Geometry:
# Collimator:
RCOL = 1 # radius of the collimator [cm]
L = 11 # length of the collimator [cm]

# Plastic target
HP = 1.5 # width plastic target [cm]
WP = 2.5 # height plastic target [cm]
LP = 3 # length plastic target [cm]
DCP = 2 # distance collimator exit - target [cm]

# Crystal detector
DPC = 38 # distance plastic target far edge - crystal detector near edge [cm]
RC = 2.54 # radius of the crystal detector [cm]
LC = 5.08 # length of the crystal detector [cm]
PHI = np.radians(0.0) # angle of the crystal detector - default [cm]

### Physics
ME = 511 # mass of electron [keV]
RE = 2.8179403262e-13 # classical electron radius [cm]
E1, E2 = 1173.240, 1332.508 # energy of the photons [keV]

### Config
E_GRID = np.linspace(1,E2+50,4*(1332+50))
MU_GRID = np.linspace(-1.0, 1.0, 1000)
THETA_GRID=np.arccos(MU_GRID)

#np.random.seed(42) # seed
N_MC = int(1e6) # number of default MC samples 
SAVE_RESULTS = False # save results or not default value