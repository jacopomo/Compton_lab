#beam_simulation.py

import argparse
import numpy as np

from mc.beam.simulation import beam_mc
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
    
    # Add an implicit -h/--help documentation spot for the arguments
    # argparse automatically generates and formats this documentation based on the 'help' strings above.

    args = parser.parse_args()

    # Pass the parsed arguments to your main simulation function
    print(f"Starting simulation with N_MC={args.num_photons}\n")
    print("-------------------------------------------------------------")
    beam_mc(n=args.num_photons, save=args.save_results, view=args.view_results)

if __name__ == "__main__":
    main()