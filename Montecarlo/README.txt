Monte Carlo Compton Scattering Simulation
=========================================

Overview
--------
This project implements a Monte Carlo simulation of Compton scattering
through a collimator, a plastic target, and a crystal detector.

It simulates photons with initial energies, tracks their interactions,
and calculates the energy deposited in the detector.

Directory Structure
-------------------
Montecarlo/
|
├─ mc/                  # Python package containing the simulation modules
│   ├─ core/
│   │   └─ simulation.py    # Main simulation routines
│   ├─ physics/
│   │   ├─ kn_sampler.py     # Klein-Nishina sampling
│   │   ├─ compton.py        # Compton scattering physics
│   │   └─ materials.py      # Material definitions
│   ├─ geometry/
│   │   ├─ surface.py        # Surfaces like disks and rectangles
│   │   └─ volume.py         # Volumes like cylinders and prisms
│   ├─ utils/
│   │   ├─ math3d.py         # Vector math utilities
│   │   └─ plotting.py       # Plotting utilities
│   └─ io/
│       └─ results.py        # Functions to save histograms and CSVs
|
├─ main.py               # Entry point for running the simulation
├─ config.py             # Global configuration (energies, grids, geometry)
├─ requirements.txt      # Python dependencies
└─ results/              # Folder created by the simulation if saving is enabled
    ├─ histograms/       # Histogram PNGs
    └─ csvs/             # CSV files with energy deposited

How to Run
----------
1. Ensure all dependencies from `requirements.txt` are installed.
   pip install -r requirements.txt

2. Run the simulation from the command line:
   python main.py -n 10000 -deg 15 -s

   Arguments:
   - -n / --num_photons : Number of photons to simulate.
   - -deg / --angle_degrees : Angle of the crystal detector in degrees.
   - -s / --save_results : Enable saving of histograms and CSVs.

Outputs
-------
- Histogram of energy deposited in the detector (PNG)  
- CSV file with energy deposited for each photon  

If saving is enabled, files are created under Montecarlo/results/:
- histograms/ : PNG histograms
- csvs/ : CSV energy spectra

Notes
-----
- The simulation is compatible with Windows and Linux.
- Default energies, geometry, and grids are configured in config.py.
- Modify N_MC or PHI in config.py for default photon number and angle.
