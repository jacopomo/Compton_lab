import numpy as np
import matplotlib.pyplot as plt

from mc.utils.plotting import plot_photon_positions
from mc.utils.math3d import unpack_stacked, generate_random_directions
from mc.geometry.surface import Rectangle, Disk
from mc.geometry.volume import RectPrism, Cylinder
from mc.core.photon import Photons
from mc.physics.kn_sampler import kn, build_kn_lut, sample_kn
from mc.config import RE, E1, E2

n = 1000
source = Disk(np.array([0,0,-10]), 2.0, np.radians(0))

posi = source.sample_unif(n)

dirs = generate_random_directions(n, (-1,1),(-1,1),(0,1))
energies = np.concatenate((np.random.normal(E1, 5, int(n/2)),np.random.normal(E2, 5, int(n/2))))

photonpool = Photons(0)
photonpool.append(posi, dirs, energies)

cyl = Cylinder(source, 10.0)

photonpool.move_to_int(cyl)
plot_photon_positions(photonpool.pos)

