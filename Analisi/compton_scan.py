#!/usr/bin/env python3

import sys
import numpy as np
import subprocess
import re
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ----------------------------
# Compton formula
# ----------------------------
def compton_energy(theta_deg, E0, me_c2):
    theta = np.deg2rad(theta_deg)
    return E0 / (1 + (E0 / me_c2) * (1 - np.cos(theta)))

# ----------------------------
# Extract fit results from triple_gaussian.py
# ----------------------------
def run_triple_gauss(script, deg, bins=120):
    cmd = [
        sys.executable,
        str(script),
        str(deg),
        "-b",
        str(bins)
    ]

    out = subprocess.check_output(cmd, text=True)

    def extract(name):
        m = re.search(rf"{name}\s*=\s*([0-9.+\-eE]+)", out)
        if m is None:
            raise RuntimeError(f"Could not find {name} in output")
        return float(m.group(1))
    
    def extract_err(name):
        m = re.search(rf"{name}\s*=\s*[0-9.+\-eE]+\s*±\s*([0-9.+\-eE]+)", out)
        if m is None:
            raise RuntimeError(f"Could not find error for {name} in output")
        return float(m.group(1))
    
    mu1 = extract("mu1")
    mu2 = extract("mu2")
    mu1_err = extract_err("mu1")
    mu2_err = extract_err("mu2")

    return mu1, mu2, mu1_err, mu2_err

# ----------------------------
# Main
# ----------------------------
def main():
    root = Path(__file__).resolve().parents[1]
    dg_script = root / "Analisi" / "tripla_gauss.py"

    data_dir = (
        root
        / "Dati"
        / "Measures"
        / "Angles"
        / "Calibrati"
    )

    angles = sorted({
        int(f.name.split("deg")[0])
        for f in data_dir.glob("*deg_*_EnergieC.txt")
    })

    mu1_list = []
    mu2_list = []
    mu1_err_list = []
    mu2_err_list = []


    print("Running triple Gaussian fits:")
    for deg in angles:
        print(f"  {deg} deg")
        mu1, mu2, mu1_err, mu2_err = run_triple_gauss(dg_script, deg)
        mu1_list.append(mu1)
        mu2_list.append(mu2)
        mu1_err_list.append(mu1_err)
        mu2_err_list.append(mu2_err)

    angles = np.array(angles)
    mu1 = np.array(mu1_list)
    mu2 = np.array(mu2_list)
    mu1_err = np.array(mu1_err_list)
    mu2_err = np.array(mu2_err_list)

    # ----------------------------
    # Fit Compton curves
    # ----------------------------
    E1 = 1180.0
    E2 = 1330.0

    popt1, pcov1 = curve_fit(
        lambda th, me: compton_energy(th, E1, me),
        angles,
        mu1,
        sigma=mu1_err,
        absolute_sigma=True,
        p0=[511.0]
    )

    popt2, pcov2 = curve_fit(
        lambda th, me: compton_energy(th, E2, me),
        angles,
        mu2,
        sigma=mu2_err,
        absolute_sigma=True,
        p0=[511.0]
    )

    me1 = popt1[0]
    me2 = popt2[0]
    me1_err = np.sqrt(pcov1[0,0])
    me2_err = np.sqrt(pcov2[0,0])

    print("\n=== COMPTON FIT RESULTS ===")
    print(f"me c^2 (1180 keV line) = {me1:.1f} ± {me1_err:.1f} keV")
    print(f"me c^2 (1330 keV line) = {me2:.1f} ± {me2_err:.1f} keV")

    # ----------------------------
    # Plot with fits and error bars
    # ----------------------------
    th_plot = np.linspace(angles.min(), angles.max(), 500)

    plt.figure(figsize=(7, 5))
    plt.errorbar(angles, mu1, yerr=mu1_err, label="μ₁ (1180 keV)", fmt="o", capsize=3, color="C0")
    plt.errorbar(angles, mu2, yerr=mu2_err, label="μ₂ (1330 keV)", fmt="s", capsize=3, color="C1")

    plt.plot(
        th_plot,
        compton_energy(th_plot, E1, me1),
        "--",
        label=f"Compton 1180 keV (me c²={me1:.0f} keV)",
        color="C0"
    )
    plt.plot(
        th_plot,
        compton_energy(th_plot, E2, me2),
        "--",
        label=f"Compton 1330 keV (me c²={me2:.0f} keV)",
        color="C1"
    )

    # plot real compton curves at 511 keV
    plt.plot(
        th_plot,
        compton_energy(th_plot, E1, 511.0),
        ":",
        label="Theoretical 1180 keV (me c²=511 keV)",
        color="black",
        alpha=0.5
    )
    plt.plot(
        th_plot,
        compton_energy(th_plot, E2, 511.0),
        ":",
        label="Theoretical 1330 keV (me c²=511 keV)",
        color="red",
        alpha=0.5
    )


    plt.xlabel("Scattering angle θ [deg]")
    plt.ylabel("Crystal energy μ [keV]")
    plt.title("Compton scattering verification")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
