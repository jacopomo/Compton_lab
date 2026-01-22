import numpy as np
import json
from pathlib import Path
import re

# --------------------------------------------------
# Paths (relative to parent directory)
# --------------------------------------------------
BASE_DIR = Path.cwd()  # run from parent parent directory
DATA_DIR = BASE_DIR / "Dati" / "Measures" / "Angles"
CALIB_FILE = BASE_DIR / "Dati" / "Calibration" / "Processed" / "calibration_table.json"
OUTPUT_DIR = DATA_DIR / "Calibrati"
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Load calibration table
# --------------------------------------------------
with open(CALIB_FILE, "r") as f:
    calib_table = json.load(f)

# --------------------------------------------------
# Regex to parse filenames like: 20deg_111225.dat
# --------------------------------------------------
filename_re = re.compile(r"(?P<angle>\d{1,2})deg_(?P<date>\d{6})\.dat")

# --------------------------------------------------
# Loop over data files
# --------------------------------------------------
for dat_file in DATA_DIR.glob("*.dat"):
    match = filename_re.match(dat_file.name)
    if not match:
        print(f"Skipping unrecognized file name: {dat_file.name}")
        continue

    angle = match.group("angle")
    date_raw = match.group("date")  # e.g. 111225
    # Convert 111225 -> 11_12_25
    day = date_raw[:2]
    month = date_raw[2:4]
    year = date_raw[4:6]
    calib_key = f"{day}_{month}_{year}"

    if calib_key not in calib_table:
        print(f"No calibration found for date {calib_key}, skipping {dat_file.name}")
        continue

    # --------------------------------------------------
    # Load counts
    # --------------------------------------------------
    counts = np.loadtxt(dat_file, dtype=int)

    if counts.size != 8192:
        raise ValueError(f"{dat_file.name} does not have 8192 rows")

    # --------------------------------------------------
    # Channel numbers
    # --------------------------------------------------
    # Assumption: channels are 1..8192
    channels = np.arange(1, counts.size + 1)

    # --------------------------------------------------
    # Apply calibration
    # --------------------------------------------------
    a, b, c = calib_table[calib_key]["coeff calib"]
    energies = a * channels**2 + b * channels + c

    # --------------------------------------------------
    # Save output
    # --------------------------------------------------
    output_data = np.column_stack((energies, counts))

    output_name = f"{angle}deg_{date_raw}_EnergieC.txt"
    output_path = OUTPUT_DIR / output_name

    np.savetxt(
        output_path,
        output_data,
        fmt="%.6e %d",
        header="Energy  Counts",
        comments=""
    )

    print(f"Saved: {output_path}")
