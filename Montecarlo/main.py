import numpy as np
import matplotlib.pyplot as plt

from mc.utils.plotting import plot_photon_positions
from mc.utils.math3d import unpack_stacked, generate_random_directions
from mc.geometry.surface import Rectangle, Disk
from mc.geometry.volume import RectPrism, Cylinder
from mc.core.photon import Photons
from mc.physics.kn_sampler import kn, build_kn_lut, sample_kn
from mc.physics.compton import compton
from mc.config import RE, E1, E2, RCOL, L
n = 1000

# Initialize photons at the source (infinite source = collimator head can be the source)

source = Disk(np.array([0,0,-L]), RCOL, 0)

pos = source.sample_unif(n)
max_dxdy = np.sin(np.arctan(2*RCOL/L)) # geometry states there is no use generating larger angles

dirs = generate_random_directions(n, (-max_dxdy,max_dxdy),(-max_dxdy,max_dxdy),(0,1)) # all interesting directions 
energies = np.concatenate((np.random.normal(E1, 5, int(n/2)),np.random.normal(E2, 5, int(n/2))))

photonpool = Photons(0)
photonpool.append(pos, dirs, energies)
plot_photon_positions(photonpool.pos)

# Have them fly through the collimator
collimator = Cylinder(source, L)

exited_base_mask = photonpool.move_to_int(collimator)
plot_photon_positions(photonpool.pos)

# If they hit the walls force compton scatter with lead
E_grid = np.linspace(1,E2+50,2*(1332+50))
theta_grid=np.linspace(0, np.pi, 360)
pdf, cdf = build_kn_lut(E_grid, theta_grid, savelut=True)

hsm = ~exited_base_mask # hit side mask
angles, weights = sample_kn(photonpool.energy[hsm], E_grid, theta_grid, cdf, theta_low=0, theta_high=np.pi/2)

photonpool.energy[hsm] = compton(photonpool.energy[hsm], angles)

photonpool.scatter_update_dirs(angles, mask=hsm)
photonpool.move_to_int(collimator, mask=hsm)

plot_photon_positions(photonpool.pos)

# Intersect them with collimator exit

# Have photons fly until plastic targer, initialize at its near end

# Photons trasverse plastic target, force one compton, when they intersect edges check if they can hit detector, if not then kill
# Now all photons are on far end of plastic

# Fly until intersection with crystal detector, note their energies
# Force at least one scatter within the detector, pe or compton
# When photons leave note their energy and calculate energy deposited
