import numpy as np
import matplotlib.pyplot as plt

from mc.utils.plotting import plot_photon_positions
from mc.utils.math3d import unpack_stacked, generate_random_directions
from mc.geometry.surface import Rectangle, Disk
from mc.core.photon import Photons
from mc.physics.kn_sampler import kn, build_kn_lut, sample_kn
from mc.config import RE, E1, E2

n = 10

# Initialize photons at the source

source = Rectangle(np.array([0,0,-10]), 4.0, 4.0, np.radians(0))

pos = source.sample_unif(n)
dirs = generate_random_directions(n, (-1,1),(-1,1),(0,1))
energies = np.concatenate((np.random.normal(E1, 5, int(n/2)),np.random.normal(E2, 5, int(n/2))))

photonpool = Photons(0)
photonpool.append(pos, dirs, energies)

# Have them fly through the collimator, if they hit the walls force compton scatter with lead
# Intersect them with collimator exit

# Have photons fly until plastic targer, initialize at its near end

# Photons trasverse plastic target, force one compton, when they intersect edges check if they can hit detector, if not then kill
# Now all photons are on far end of plastic

# Fly until intersection with crystal detector, note their energies
# Force at least one scatter within the detector, pe or compton
# When photons leave note their energy and calculate energy deposited
