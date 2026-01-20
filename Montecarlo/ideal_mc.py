#ideal_mc.py

import argparse
import numpy as np

from mc.ideal_mc.simulation import imc
from mc.config import N_MC


def main():
    """
    Main entry point for the Monte Carlo Simulation.
    Parses command line arguments and runs the simulation.
    """
    parser = argparse.ArgumentParser(
        description="Run the Compton Scattering Monte Carlo simulation.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Argument for changing photons' number
    parser.add_argument(
        '-n', 
        '--num_photons', 
        type=int, 
        default=N_MC,
        help=f"Set the number of Monte Carlo photons to simulate. (Default: {N_MC})"
    )

    # Argument for changing NaI angle
    parser.add_argument(
        '-deg', 
        '--degree', 
        type=float, 
        default=0.0,
        help=f"Set the NaI position's angle in deg. (Default: 0.0 deg)"
    )
    
    # Argument for whether to save or not
    parser.add_argument(
        '-s', 
        '--save_results', 
        # --- Change type=bool to action='store_true' ---
        action='store_true', 
        default=False,
        help=f"Enable saving of final histogram CSVs in Montecarlo/results. (Default: False)"
    )

    # Argument for whether to view or not graphs
    parser.add_argument(
        '-v', 
        '--view_results', 
        # --- Change type=bool to action='store_true' ---
        action='store_true', 
        default=False,
        help=f"Enable visualize of final histogram. (Default: False)"
    )
    
    # Argument for debugging
    parser.add_argument(
        '-d',
        '--debug_flag',
        action='store_true',
        default=False,
        help="Enable other histograms normally not visible."
    )
    
    # Add an implicit -h/--help documentation spot for the arguments
    # argparse automatically generates and formats this documentation based on the 'help' strings above.

    args = parser.parse_args()

    # Pass the parsed arguments to your main simulation function
    print(f"Starting simulation with N={args.num_photons}\n")
    print("-------------------------------------------------------------")
    imc(n=args.num_photons, PHI=args.degree, view=args.view_results, debug=args.debug_flag)

if __name__ == "__main__":
    main()