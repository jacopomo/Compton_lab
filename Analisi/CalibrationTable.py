import json
import numpy as np
import os
import argparse

from CalibrationCurve import calibration

def calibrationTable(bar_printing=False, fits=False, visualizzare=False):
    _dir = "Dati"
    _subdir = "Calibration"
    base = os.getcwd()
    path = os.path.join(base, "..", _dir, _subdir)

    out_path = os.path.join(base, "..", _dir, _subdir, "Processed")

    folders = [
        f for f in os.listdir(path)
        if os.path.isdir(os.path.join(path, f)) and f not in ["Config", "Processed"]
    ]

    table_calib = {}

    for data in folders:

        coeff, r2, coeff_risol, s2 = calibration(day=data, bar_printing=bar_printing, fits=fits, bar_visualizzare = (not visualizzare))
        table_calib[data] = {"coeff calib": coeff.tolist(), "r2": r2, "coeff risol": coeff_risol.tolist(), "s2": s2}
        '''
        except:
            print(f"La cartella '{data}' non contiene i file giusti per eseguire il codice!\n")
        '''
    with open(os.path.join(out_path, "calibration_table.json"), "w", encoding="utf-8") as f:
        json.dump(table_calib, f, ensure_ascii=False, indent=4)





def main():
    """
    Main entry point for the calibration of one day.
    Parses command line arguments and runs the calibration.
    """
    parser = argparse.ArgumentParser(
        description="Run the calibration analysis in a specific date.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '-f', 
        '--fits', 
        action='store_true',
        default=False,
        help=f"View all graphs from fits."
    )

    parser.add_argument(
        '-p', 
        '--print', 
        action='store_true',
        default=False,
        help=f"Disabe printing console's messages."
    )

    parser.add_argument(
        '-v', 
        '--view', 
        action='store_true',
        default=False,
        help=f"View final calibration curves."
    )

    args = parser.parse_args()

    # Pass the parsed arguments to your main simulation function
    print(f"Starting write calibration table.\n")
    print("======================================================")


    calibrationTable(bar_printing=args.print, fits=args.fits, visualizzare=args.view)

if __name__ == "__main__":
    main()

