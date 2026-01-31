# mc/core/simulation.py

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
import mc.utils.plotting as pl

from mc.utils.plotting import plot_photon_positions
from mc.utils.math3d import generate_random_directions, lift_mask, unpack_stacked
from mc.geometry.surface import Rectangle, Disk
from mc.geometry.volume import RectPrism, Cylinder
from mc.core.photon import Photons
from mc.physics.kn_sampler import sample_kn, PDF, CDF
from mc.physics.compton import compton
from mc.physics.materials import Material
from mc.io.results import save_csv, save_histogram
from mc.config import RE, E1, E2, RCOL, L, WP, LP, HP, DCP, DPC, LC, RC, MU_GRID, E_GRID, ME
def cmc(n, phi, save):
    start = time.time()

    # Initialize photons at the source (infinite source = collimator head can be the source)
    source = Disk(np.array([0,0,-L]), RCOL, 0)

    pos = source.sample_unif(n)
    max_dxdy = np.sin(np.arctan(2*RCOL/L)) # geometry states there is no use generating larger angles

    dirs = generate_random_directions(n, (-max_dxdy,max_dxdy),(-max_dxdy,max_dxdy),(0,1)) # all interesting directions 
    energies = np.concatenate((np.random.normal(E1, 5, int(n/2)),np.random.normal(E2, 5, int(n/2))))

    photonpool = Photons(0)
    photonpool.append(pos, dirs, energies)

    #plot_photon_positions(photonpool.pos)
    print(f"{n} photons initialized at the source!")

    # Have them fly through the collimator
    collimator = Cylinder(source, L)

    exited_base_mask = photonpool.move_to_int(collimator)

    # If they hit the walls force compton scatter with lead

    hsm = ~exited_base_mask # hit side mask
    angles, weights = sample_kn(photonpool.energy[hsm], E_GRID, MU_GRID, CDF)
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

    #plot_photon_positions(photonpool.pos)
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

    photonpool.E_depo_in_plastic = photonpool.energy.copy() # initialize energy deposit tracking
    E_th_plastic = 0 # np.random.normal(60.0, 15.0, len(photonpool.alive))
    exit_base_mask, mus = photonpool.force_one_scatter_moveto(plastic, E_th=E_th_plastic)
    exit_front_mask = photonpool.pos[:,2] > DCP-0.1+LP-1e-3
    hit_plastic_front_mask = exit_base_mask & exit_front_mask
    # Only store scatter angles for photons that successfully exit the front
    photonpool.scatter_in_plastic[hit_plastic_front_mask] = mus[hit_plastic_front_mask]

    photonpool.alive[~hit_plastic_front_mask] = False
    photonpool.compact() # remove dead
    photonpool.E_depo_in_plastic = photonpool.E_depo_in_plastic - photonpool.energy
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
    cryface = Disk(np.array([0, R*np.sin(phi), R*np.cos(phi)]), RC, phi)
    crystal = Cylinder(cryface, LC, material=NaI)
    hit_crystal_mask = photonpool.moveto_int_disk(cryface)

    photonpool.alive[~hit_crystal_mask] = False

    photonpool.compact()
    alive_n = photonpool.alive.sum() 
    print(f"{alive_n} photons have hit the crystal ({round(alive_n*100/n,2)}% of original)")

    #plot_photon_positions(photonpool.pos)
    show_E_crystalin_graph = False
    if show_E_crystalin_graph:
        plt.hist(photonpool.energy, bins=80, histtype="step", weights=photonpool.weight)
        plt.title(f"Energy spectrum of photons that enter the crystal ({round(np.degrees(phi),1)} degrees)")
        plt.xlabel("Energy [keV]")
        plt.ylabel("Counts")
        plt.show()

    n_split = int(round(n/alive_n)) # number to multiply by to get back to n
    photonpool.split(n_split)
    print(f"\nBefore calculating what happens in the crystal I will multiply them by {n_split} to get back to {n} photons")

    # Force at least one scatter within the detector, pe or compton
    # When photons leave note their energy and calculate energy deposited
    analyzed_n = photonpool.alive.sum()
    E_th_crystal = 0 #np.random.normal(700.0, 25.0, len(photonpool.alive))
    E_deposited = photonpool.force_first_then_transport(crystal, E_th=E_th_crystal)
    
    alive_mask = photonpool.alive # photons who passed the threshold
    E_deposited = E_deposited[alive_mask]
    weightss = photonpool.weight[alive_mask]
 
    print(f"{analyzed_n} photons have been analyzed ({round(analyzed_n*100/n,2)}% of original)")
    print(f"{len(E_deposited)} photons surpassed the threshold and were seen ({round(len(E_deposited)*100/n,2)}% of original)")

    end = time.time()
    print("-------------------------------------------------------------\n")
    print(f'Runtime: {round(end - start,2)}s')
    plt.hist(E_deposited, bins=80, histtype="step", weights=weightss)
    plt.title(f"Energy deposited in the crystal ({round(np.degrees(phi),1)} degrees)")
    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts")
    plt.show()


    theta_final = np.arccos(photonpool.scatter_in_plastic[alive_mask])
    weights_corrected_final = weightss / np.sin(theta_final + 1e-10)
    plt.hist(np.degrees(theta_final), bins=80, histtype="step", weights=weights_corrected_final)
    plt.title(f"Distribution of scattering in plastic ({round(np.degrees(phi),1)} degrees)")
    plt.xlabel("Scattering angle [degrees]")
    plt.ylabel("Counts")
    print(f"Average scattering angle in plastic: {round(np.degrees(np.average(theta_final, weights=weights_corrected_final)),2)} degrees, STD: {round(np.degrees(np.std(theta_final, ddof=1)),2)} degrees")
    plt.show()


    lost_energies = photonpool.E_depo_in_plastic[alive_mask]
    H, xedges, yedges, _, _ = pl.hist2d(E_deposited, lost_energies, bins=1000, weights=weightss, title="Istogramma finale dell'energia depositata nel cristallo vs quella nel plastico", cmap="viridis")

    if save:
        degPHI = str(int(np.degrees(phi)))
        file_name = degPHI + 'deg.h5'

        path_dir = os.path.join('results', '2d_histograms')
        os.makedirs(path_dir, exist_ok=True)

        path_file = os.path.join(path_dir, file_name)

        with h5py.File(path_file, 'w') as f:
            print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("\nOutput file .h5 saved in Montecarlo/results.\n")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
            f.create_dataset('H', data=H)
            f.create_dataset('xedges', data=xedges)
            f.create_dataset('yedges', data=yedges)
