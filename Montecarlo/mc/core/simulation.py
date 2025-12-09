# mc/core/simulation.py

import numpy as np
import matplotlib.pyplot as plt

from mc.utils.plotting import plot_photon_positions
from mc.utils.math3d import generate_random_directions, lift_mask
from mc.geometry.surface import Rectangle, Disk
from mc.geometry.volume import RectPrism, Cylinder
from mc.core.photon import Photons
from mc.physics.kn_sampler import build_kn_lut, sample_kn
from mc.physics.compton import compton
from mc.config import RE, E1, E2, RCOL, L
def mc():
    n = int(1e5)
    np.random.seed(42) # Seed

    # Initialize photons at the source (infinite source = collimator head can be the source)

    source = Disk(np.array([0,0,-L]), RCOL, 0)

    pos = source.sample_unif(n)
    max_dxdy = np.sin(np.arctan(2*RCOL/L)) # geometry states there is no use generating larger angles

    dirs = generate_random_directions(n, (-max_dxdy,max_dxdy),(-max_dxdy,max_dxdy),(0,1)) # all interesting directions 
    energies = np.concatenate((np.random.normal(E1, 5, int(n/2)),np.random.normal(E2, 5, int(n/2))))

    photonpool = Photons(0)
    photonpool.append(pos, dirs, energies)
    print("Photons initialized at the source!")

    # Have them fly through the collimator
    collimator = Cylinder(source, L)

    exited_base_mask = photonpool.move_to_int(collimator)
    print(f"Photons moved to collimator, there were {exited_base_mask.sum()} exits")

    # If they hit the walls force compton scatter with lead
    E_grid = np.linspace(1,E2+50,2*(1332+50))
    theta_grid=np.linspace(0, np.pi, 360)
    pdf, cdf = build_kn_lut(E_grid, theta_grid, savelut=True)

    hsm = ~exited_base_mask # hit side mask
    angles, weights = sample_kn(photonpool.energy[hsm], E_grid, theta_grid, cdf)
    photonpool.energy[hsm] = compton(photonpool.energy[hsm], angles)

    photonpool.scatter_update_dirs(angles, mask=hsm)
    new_exited_base_mask = photonpool.move_to_int(collimator, mask=hsm)
    
    # Intersect them with collimator exit
    new_hsm_small = ~new_exited_base_mask # new hit side mask (not len N)
    new_hsm = lift_mask(new_hsm_small, hsm) # project it to len N
    exited_back_mask = photonpool.pos[:,2] < -0.1
    hit_side_or_back = new_hsm | exited_back_mask
    photonpool.alive[hit_side_or_back] = False

    print(f"then there were {new_exited_base_mask.sum()} more hits")
    print(f"Now there are {photonpool.alive.sum()} photons that have exited the collimator ({round(photonpool.alive.sum()/n,4)*100}%)")
    photonpool.compact() # remove dead

    show_E_collimator_graph = True
    if show_E_collimator_graph:
        plt.hist(photonpool.energy, bins=60, histtype="step")
        plt.title("Energy spectrum of photons that exit the collimator")
        plt.xlabel("Energy [keV]")
        plt.ylabel("Counts")
        plt.show()

    # Have photons fly until plastic target, initialize at its near end

    # Photons trasverse plastic target, force one compton, when they intersect edges check if they can hit detector, if not then kill
    # Now all photons are on far end of plastic

    # Fly until intersection with crystal detector, note their energies
    # Force at least one scatter within the detector, pe or compton
    # When photons leave note their energy and calculate energy deposited
