import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
import os

from mc.geometry.surface import Disk, Rectangle
from mc.geometry.volume import RectPrism, Cylinder
from mc.core.photon import Photons
from mc.physics.materials import Material
from mc.physics.compton import compton
from mc.physics.kn_sampler import kn
import mc.utils.plotting as pl

from mc.config import RE, E1, E2, RCOL, L, WP, LP, HP, DCP, DPC, LC, RC, MU_GRID, E_GRID

def imc(n, PHI, view, debug=False):
    PHI = np.radians(PHI)

    start = time.time()
    print("Start inizialitazion...")

    #Initialization of surfaces and valumes
    base_collimator = Disk(np.array([0,0,-L-DCP-(HP/2)]), RCOL, 0)
    exit_collimator = Disk(np.array([0,0,-DCP-(LP/2)]), RCOL, 0)

    volume_collimator = Cylinder(base_collimator, L)

    base_plastic = Rectangle(np.array([0,0,-LP/2]), WP, HP, 0)
    C = Material("C")
    volume_plastic = RectPrism(base_plastic, LP, material=C)

    #Initialization of photons
    pos_base = base_collimator.sample_unif(n)
    pos_exit = exit_collimator.sample_unif(n)

    dirs_non_norm = pos_exit - pos_base
    norms = np.linalg.norm(dirs_non_norm, axis=1)
    dirs = dirs_non_norm / norms[:, None]

    energies = np.random.choice([E1, E2], n)

    photons = Photons(0)
    photons.append(pos_exit, dirs, energies)

    current_time = time.time()
    print(f"Finish inizialization ({current_time-start:.3} s)")
    print("==================================================")
    print("Start simulation...")

    if debug:
        pl.plot_entry_exit(pos_base, pos_exit)
        pl.plot_histogram(energies, bins=1000, title="Energia iniziale dei fotoni uscenti dal collimatore.")

    #Propagation of photons in the plastic
    hit_plastic_mask = photons.moveto_int_rect(base_plastic)
    photons.alive[~hit_plastic_mask] = False
    photons.compact()
    new_n = photons.alive.sum()

    print(f"\nN = {new_n} ({new_n/n*100:.1f} %) have hit the plastic\n")
    n = new_n

    int_pos, w = photons._force_first_compton_only_pos(volume_plastic)

    current_time = time.time()
    print(f"\nGenerated interaction point in the plastic... ({current_time-start:.3f} s)")

    if debug:
        pl.plot_photon_positions(int_pos, N_max=1e3)
        pl.plot_histogram_Nm_overlay(int_pos, bins=100)

    #Inizialization of NaI's surface and final directions
    R = DPC + LP/2
    base_crystal = Disk(np.array([0, R*np.sin(PHI), R*np.cos(PHI)]), RC, PHI)

    NaI = Material("NaI")
    volume_crystal = Cylinder(base_crystal, LC, material=NaI)

    final_pos = base_crystal.sample_unif(n)

    if debug:
        pl.plot_photon_positions(final_pos, N_max=1e3)

    final_dirs_non_norm = final_pos - int_pos
    norms = np.linalg.norm(final_dirs_non_norm, axis=1)
    final_dirs = final_dirs_non_norm / norms[:, None]


    #Calculate the Compton energies and the weight
    cos_theta = np.sum(dirs * final_dirs, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    new_n = len(cos_theta)

    current_time = time.time()
    print(f"\nAfter clipping cos_theta remain N = {new_n} photons ({new_n/n*100:.1f} %). ({current_time-start:.3f} s)\n")
    n = new_n

    final_energies = compton(energies, cos_theta)
    lost_energies = energies - final_energies
    _weight = kn(energies, cos_theta)
    weight = (w * _weight/np.max(w * _weight))

    if debug:
        pl.compare_weighted_unweighted_histogram(np.degrees(np.arccos(cos_theta)),weight, bins=100)
        pl.scatter_xy(cos_theta, final_energies, lines=[lambda x, mu=mu: compton(mu, x) for mu in [E1, E2]], N_max=1000, title="Scatter di energia vs angolo per i fotoni uscenti dal plastico.")
        pl.scatter_xy(cos_theta, _weight, lines=[lambda x, mu=mu: kn(mu, x) for mu in [E1, E2]], N_max=1000, title="Scatter di peso vs angolo.")

    if view or debug:
        pl.plot_histogram(final_energies, weights=weight, bins=300, title="Spettro di energia dei fotoni incidenti sul NaI")

    eq_n = weight.sum()

    current_time = time.time()
    print(f"The alive photons are equivalent to N = {eq_n:.1f} photons ({eq_n/n*100:.1f} %). ({current_time-start:.3f} s)")

    #Updating energies and directions for photons
    photons = Photons(0)
    photons.append(final_pos, final_dirs, final_energies)

    #Propagation in the crystal
    energies_depo = photons.force_first_then_transport(volume_crystal, 0)
    
    w = photons.weight.copy()
    final_weight = w * weight
    final_weight = (final_weight/np.max(final_weight))

    eq_n = final_weight.sum()

    current_time = time.time()
    print(f"Finish simulation. The alive photons are equivalent to N = {eq_n:.1f} photons ({eq_n/n*100:.1f} %). ({current_time-start:.3f} s)")

    if view or debug:
        pl.hist2d(energies_depo, lost_energies, bins=200, weights=final_weight, title="Istogramma finale dell'energia depositata nel cristallo vs quella nel plastico", cmap="viridis")

    if view or debug:
        pl.plot_histogram(energies_depo, weights=final_weight, bins=300, title="Spettro dell'energia depositata nel NaI")