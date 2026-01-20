
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from mc.geometry.surface import Disk
from mc.geometry.volume import Cylinder
from mc.core.photon import Photons
from mc.utils.math3d import generate_random_directions
from mc.utils.plotting import plot_entry_exit, hist_radial_vs_energy

from mc.config import RCOL, L, E1, E2

def beam_mc(n, save, view):
    start = time.time()
    print("Start inizialization...")
    
    #Inizialize the collimator valume and surfaces
    base_disk = Disk(np.array([0,0,-L]), RCOL, 0)
    volume_collimator = Cylinder(base_disk, L)

    #Inizialize the photonpool and generate uniformilly photons on th base_disk
    uni_pos = base_disk.sample_unif(n)
    max_dxdy = np.sin(np.arctan(2*RCOL/L)) # geometry states there is no use generating larger angles

    dirs = generate_random_directions(n, (-max_dxdy,max_dxdy),(-max_dxdy,max_dxdy),(0,1)) # all interesting directions 
    energies = np.concatenate((np.random.normal(E1, 5, int(n/2)),np.random.normal(E2, 5, int(n/2))))

    photonpool = Photons(0)
    photonpool.append(uni_pos, dirs, energies)

    finish_inizialization = time.time()
    print(f"Finish inizialization ({finish_inizialization-start:.6f} s)")
    print("===========================================================")
    print(f"Start simulation...")

    #I exclude photons if they hit collimator's side
    exited_base_mask = photonpool.move_to_int(volume_collimator)
    photonpool.alive[~exited_base_mask] = False
    photonpool.compact()

    initial_pos = uni_pos[exited_base_mask]
    final_pos = photonpool.pos

    direc = photonpool.direc
    norm = base_disk.normal

    cos = np.sum(direc * norm, axis=1)/np.linalg.norm(direc, axis=1)


    fraction_alive = len(initial_pos)/n * 100

    finish_simulation = time.time()
    print(f"Finish simulation ({finish_simulation-start:.6f} s)")
    print("===========================================================\n")
    print(f"Percentage of photons alive: {fraction_alive:.1f} %")

    histo_name = "beam_simulation.png"
    _base = os.getcwd()
    _dir = "results"
    _subdir = "histograms" 
    histo_path = os.path.join(_base, _dir, _subdir, histo_name)

    if view or save:
        fig, ax = plot_entry_exit(initial_pos, final_pos, RCOL+0.2)
        hist_radial_vs_energy(final_pos, cos, center=np.array([0,0,0]), normal=base_disk.normal)
    if save:
        fig.savefig(histo_path, dpi = 900)