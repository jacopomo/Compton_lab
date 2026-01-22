#main.py

import argparse
import numpy as np

from mc.core.simulation import cmc
from mc.config import N_MC, PHI, SAVE_RESULTS


def main():
    """
    Main entry point for the Monte Carlo Simulation.
    Parses command line arguments and runs the simulation.
    """
    parser = argparse.ArgumentParser(
        description="Run the Compton Scattering Monte Carlo simulation.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '-n', 
        '--num_photons', 
        type=int, 
        default=N_MC,
        help=f"Set the number of Monte Carlo photons to simulate. (Default: {N_MC})"
    )

    # Argument for changing the angle constant
    parser.add_argument(
        '-deg', 
        '--angle_degrees', 
        type=float, 
        default=np.round(np.degrees(PHI),1),
        help=f"Set the initial angle in degrees for the simulation setup. (Default: {np.round(np.degrees(PHI),1)})"
        )
    
     # Argument for whether to save or not
    parser.add_argument(
        '-s', 
        '--save_results', 
        # --- Change type=bool to action='store_true' ---
        action='store_true', 
        default=SAVE_RESULTS,
        help=f"Enable saving of final histogram and energy spectrum CSVs in Montecarlo/results. (Default: {SAVE_RESULTS})"
    )
    
    # Add an implicit -h/--help documentation spot for the arguments
    # argparse automatically generates and formats this documentation based on the 'help' strings above.

    args = parser.parse_args()

    # Pass the parsed arguments to your main simulation function
    print(f"Starting simulation with N_MC={args.num_photons}, Angle={args.angle_degrees} deg\n")
    print("-------------------------------------------------------------")
    cmc(n=args.num_photons, phi=np.radians(args.angle_degrees), save=args.save_results)

if __name__ == "__main__":
    main()
