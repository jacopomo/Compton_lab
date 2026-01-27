#!/usr/bin/env python3

import sys
import numpy as np
import subprocess
import re
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

E1 = 1180.0
E2 = 1330.0

# ----------------------------
# Compton formula
# ----------------------------
def compton_energy(theta_deg, E0, me_c2):
    theta = np.deg2rad(theta_deg)
    return E0 / (1 + (E0 / me_c2) * (1 - np.cos(theta)))

def dcompton_dtheta(theta_deg, E0, me_c2):
    theta = np.deg2rad(theta_deg)
    num = E0 * (E0 / me_c2) * np.sin(theta)
    den = (1 + (E0 / me_c2) * (1 - np.cos(theta)))**2
    return num / den * (np.pi / 180.0)  # keV / deg

# ----------------------------
# Extract fit results from gauss_fit.py
# ----------------------------
def gauss_fit(script, deg, filename, bins=120, trip=False):
    cmd = [
        sys.executable,
        str(script),
        filename,
        "-b",
        str(bins),
    ]

    if trip:
        cmd += ["-t"]

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

    # Effective uncertainties including theta errors
    sigma_mu1_eff = np.sqrt(
        mu1_err**2 +
        (dcompton_dtheta(angles, E1, 511.0) * theta_err)**2
    )

    sigma_mu2_eff = np.sqrt(
        mu2_err**2 +
        (dcompton_dtheta(angles, E2, 511.0) * theta_err)**2
    )


    return mu1, mu2, mu1_err, mu2_err, sigma_mu1_eff, sigma_mu2_eff

# ----------------------------
# Main
# ----------------------------

# ----------------------------
# DEFS
# ----------------------------
angles = np.array([0, 10, 15, 15, 20, 25, 30, 30, 35, 40], dtype=float)

files = [
    "0deg_261125_EnergieC.txt",
    "10deg_041225_EnergieC.txt",
    "15deg_111225_EnergieC.txt",
    "15deg_251125_EnergieC.txt",
    "20deg_251125_EnergieC.txt",
    "25deg_261125_EnergieC.txt",
    "30deg_021225_EnergieC.txt",
    "30deg_201125_EnergieC.txt",
    "35deg_271125_EnergieC.txt",
    "40deg_031225_EnergieC.txt",
]

theta_err = np.array([
    5.34,  # 0 deg
    5.73,  # 10
    5.97,  # 15
    5.97,  # 15
    5.99,  # 20
    6.13,  # 25
    6.21,  # 30
    6.21,  # 30
    6.08,  # 35
    6.12,  # 40
])

theta_err /= 2.0  # use half-width error

use_triple = np.array([
    True,  # 0 deg
    True,  # 10
    True,   # 15
    True,   # 15
    True,   # 20
    False,  # 25
    False,  # 30
    False,  # 30
    False,  # 35
    False,  # 40
], dtype=bool)

def main():
    root = Path(__file__).resolve().parents[1]
    dg_script = root / "Analisi" / "gauss_fit.py"

    data_dir = (
        root
        / "Dati"
        / "Measures"
        / "Angles"
        / "Calibrati"
    )
    

    mu1_list = []
    mu2_list = []
    mu1_err_list = []
    mu2_err_list = []



    print("Running Gaussian fits:")
    for deg, fname, trip in zip(angles, files, use_triple):
        print(f"  {deg} deg | Triple Gaussian: {trip}")
        mu1, mu2, mu1_err, mu2_err, sigma_mu1_eff, sigma_mu2_eff = gauss_fit(dg_script, deg, str(data_dir / fname), trip=trip)
        mu1_list.append(mu1)
        mu2_list.append(mu2)
        mu1_err_list.append(mu1_err)
        mu2_err_list.append(mu2_err)

    mu1 = np.array(mu1_list)
    mu2 = np.array(mu2_list)
    mu1_err = np.array(mu1_err_list)
    mu2_err = np.array(mu2_err_list)


    # ----------------------------
    # Fit Compton curves
    # ----------------------------

    popt1, pcov1 = curve_fit(
        lambda th, me: compton_energy(th, E1, me),
        angles,
        mu1,
        sigma=sigma_mu1_eff,
        absolute_sigma=True,
        p0=[511.0]
    )

    popt2, pcov2 = curve_fit(
        lambda th, me: compton_energy(th, E2, me),
        angles,
        mu2,
        sigma=sigma_mu2_eff,
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
    plt.errorbar(angles, mu1, xerr=theta_err, yerr=mu1_err, label="μ₁ (1180 keV)", fmt="o", capsize=3, color="C0")
    plt.errorbar(angles, mu2, xerr=theta_err, yerr=mu2_err, label="μ₂ (1330 keV)", fmt="s", capsize=3, color="C1")

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
