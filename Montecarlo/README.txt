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
│
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
│
├─ main.py               # Entry point for running the simulation
├─ config.py             # Global configuration (energies, grids, geometry)
├─ requirements.txt      # Python dependencies
└─ results/              # Folder created by the simulation if saving is enabled
    ├─ histograms/       # Histogram PNGs
    └─ csvs/             # CSV files with energy deposited

How to Run
----------
1. Ensure all dependencies from `requirements.txt` are installed.
   ```bash
   pip install -r requirements.txt
