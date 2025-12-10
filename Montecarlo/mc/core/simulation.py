# mc/core/simulation.py

import numpy as np
import matplotlib.pyplot as plt
import time

from mc.utils.plotting import plot_photon_positions
from mc.utils.math3d import generate_random_directions, lift_mask, unpack_stacked
from mc.geometry.surface import Rectangle, Disk
from mc.geometry.volume import RectPrism, Cylinder
from mc.core.photon import Photons
from mc.physics.kn_sampler import sample_kn, PDF, CDF, THETA_GRID, E_GRID
from mc.physics.compton import compton
from mc.physics.materials import Material
from mc.config import RE, E1, E2, RCOL, L, WP, LP, HP, DCP, DPC, LC, RC, N_MC, PHI
def cmc():
    start = time.time()
    n = N_MC

    # Initialize photons at the source (infinite source = collimator head can be the source)
    source = Disk(np.array([0,0,-L]), RCOL, 0)

    pos = source.sample_unif(n)
    max_dxdy = np.sin(np.arctan(2*RCOL/L)) # geometry states there is no use generating larger angles

    dirs = generate_random_directions(n, (-max_dxdy,max_dxdy),(-max_dxdy,max_dxdy),(0,1)) # all interesting directions 
    energies = np.concatenate((np.random.normal(E1, 5, int(n/2)),np.random.normal(E2, 5, int(n/2))))

    photonpool = Photons(0)
    photonpool.append(pos, dirs, energies)
    print(f"{n} photons initialized at the source!")

    # Have them fly through the collimator
    collimator = Cylinder(source, L)

    exited_base_mask = photonpool.move_to_int(collimator)

    # If they hit the walls force compton scatter with lead

    hsm = ~exited_base_mask # hit side mask
    angles, weights = sample_kn(photonpool.energy[hsm], E_GRID, THETA_GRID, CDF)
    photonpool.energy[hsm] = compton(photonpool.energy[hsm], angles)
    photonpool.weight[hsm] = weights * 0.21 # PLACEHOLDER 
    
    photonpool.scatter_update_dirs(angles, mask=hsm)
    new_exited_base_mask = photonpool.move_to_int(collimator, mask=hsm)
    
    # Intersect them with collimator exit
    new_hsm_small = ~new_exited_base_mask # new hit side mask (not len N)
    new_hsm = lift_mask(new_hsm_small, hsm) # project it to len N
    exited_back_mask = photonpool.pos[:,2] < -0.1
    hit_side_or_back = new_hsm | exited_back_mask

    photonpool.alive[hit_side_or_back] = False
    
    photonpool.compact() # remove dead
    alive_n = photonpool.alive.sum() 
    print(f"{alive_n} photons have exited the collimator ({round(alive_n*100/n,2)}% of original)")

    show_E_collimator_graph = False
    if show_E_collimator_graph:
        plt.hist(photonpool.energy, bins=80, histtype="step", weights=photonpool.weight)
        plt.title("Energy spectrum of photons that exit the collimator")
        plt.xlabel("Energy [keV]")
        plt.ylabel("Counts")
        plt.show()

    # Have photons fly until plastic target, initialize at its near end
    plasticface = Rectangle(np.array([0,0,DCP]), WP, HP, 0)

    hit_plastic_mask = photonpool.moveto_int_rect(plasticface)
    photonpool.alive[~hit_plastic_mask] = False

    photonpool.compact() # remove dead
    alive_n = photonpool.alive.sum()
    print(f"{alive_n} photons have hit the plastic ({round(alive_n*100/n,2)}% of original)")

    #plot_photon_positions(photonpool.pos) # photon positions on the target
    n_split = int(round(n/alive_n)) # number to multiply by to get back to n
    photonpool.split(n_split)
    print(f"\nBefore calculating what happens in the plastic I will multiply them by {n_split} to get back to {n} photons")

    C = Material("C")
    pface = Rectangle(np.array([0,0,DCP-0.1]), WP, HP, 0)
    plastic = RectPrism(pface, LP+0.1, material=C)
    #plot_photon_positions(photonpool.pos) # Where they exit the plastic

    E_th_plastic = np.random.normal(85.0, 5.0, len(photonpool.alive))
    exit_base_mask = photonpool.force_one_scatter_moveto(plastic, E_th=E_th_plastic)
    exit_front_mask = photonpool.pos[:,2] > DCP-0.1+LP-1e-3
    hit_plastic_front_mask = exit_base_mask & exit_front_mask
    photonpool.alive[~hit_plastic_front_mask] = False
    photonpool.compact() # remove dead
    alive_n = photonpool.alive.sum() 
    print(f"{alive_n} photons have exited the front face of the plastic ({round(alive_n*100/n,2)}% of original)")
    #plot_photon_positions(photonpool.pos)

    show_E_plastic_graph = False
    if show_E_plastic_graph:
        plt.hist(photonpool.energy, bins=80, histtype="step", weights=photonpool.weight)
        plt.title("Energy spectrum of photons that exit the plastic")
        plt.xlabel("Energy [keV]")
        plt.ylabel("Counts")
        plt.show()
    # Photons trasverse plastic target, force one compton with minimum angle such that it 
    # deposits at least 85keV
    # when they intersect edges check if they can hit detector, if not then kill
    # Now all photons are on far end of plastic


    # Fly until intersection with crystal detector, note their energies
    R = DCP + LP + DPC
    NaI = Material("NaI")
    cryface = Disk(np.array([0, R*np.sin(PHI), R*np.cos(PHI)]), RC, PHI)
    crystal = Cylinder(cryface, LC, material=NaI)
    hit_crystal_mask = photonpool.moveto_int_disk(cryface)

    photonpool.alive[~hit_crystal_mask] = False

    photonpool.compact()
    alive_n = photonpool.alive.sum() 
    print(f"{alive_n} photons have hit the crystal ({round(alive_n*100/n,2)}% of original)")

    #plot_photon_positions(photonpool.pos)
    show_E_crystalin_graph = True
    if show_E_crystalin_graph:
        plt.hist(photonpool.energy, bins=80, histtype="step", weights=photonpool.weight)
        plt.title(f"Energy spectrum of photons that enter the crystal ({round(np.degrees(PHI),1)} degrees)")
        plt.xlabel("Energy [keV]")
        plt.ylabel("Counts")
        plt.show()

    n_split = int(round(n/alive_n)) # number to multiply by to get back to n
    photonpool.split(n_split)
    print(f"\nBefore calculating what happens in the crystal I will multiply them by {n_split} to get back to {n} photons")

    # Force at least one scatter within the detector, pe or compton
    # When photons leave note their energy and calculate energy deposited

    E_th_crystal = np.random.normal(750.0, 5.0, len(photonpool.alive))
    E_initials = photonpool.energy.copy()
    E_finals = photonpool.force_first_then_transport(crystal, E_th=E_th_crystal)
    E_deposited = E_initials - E_finals
    analyzed_n = len(E_deposited) 
    print(f"{analyzed_n} photons have been analyzed ({round(analyzed_n*100/n,2)}% of original)")

    end = time.time()
    print("\n--------")
    print(f'Tempo impiegato: {round(end - start,2)}s')

    plt.hist(E_deposited, bins=80, histtype="step", weights=photonpool.weight)
    plt.title(f"Energy deposited in the crystal ({round(np.degrees(PHI),1)} degrees)")
    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts")
    plt.show()

