import json
import numpy as np
import os

import CalibrationCurve as c

_dir = "Dati"
_subdir = "Calibration"
base = os.getcwd()
path = os.path.join(base, _dir, _subdir)

out_path = os.path.join(base, _dir, _subdir, "Processed")

folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

table_calib = {}

for data in folders:
    try:
        m, q, r2 = c.Calibration(os.path.join(path, data))
        table_calib[data] = {"m": m, "q": q, "r2": r2}
    except:
        print(f"La cartella '{data}' non contiene i file giusti per eseguire il codice!\n")
    
with open(os.path.join(out_path, "calibration_table.json"), "w", encoding="utf-8") as f:
    json.dump(table_calib, f, ensure_ascii=False, indent=4)


