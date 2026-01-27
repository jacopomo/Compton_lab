#!/usr/bin/env python3

import sys
import numpy as np
import subprocess
import re
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import json

E1 = 1332.0
E2 = 1173.0
ME_C2 = 511.0  # keV

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
def gauss_fit(script, filepath, bins=120):
    cmd = [
        sys.executable,
        str(script),
        filepath,
        "-b",
        str(bins),
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
    
    fit_script = root / "Analisi" / "gauss_fit.py"
    data_dir = root / "Dati" / "Measures" / "Angles" / "Calibrati"
    config_path = root / "Analisi" / "fit_config.json"

    with open(config_path) as f:
        config = json.load(f)

    angles = []
    theta_err = []
    mu1_list = []
    mu2_list = []
    mu1_err_list = []
    mu2_err_list = []



    print("Running Gaussian fits:")
    for fname, cfg in config.items():
        deg = cfg["degree"]
        sigma_theta = cfg["theta_error"]
        use_triple = (cfg.get("model", "double") == "triple")

        print(f"  {deg:>3} deg | triple={use_triple} | {fname}")

        mu1, mu2, mu1_err, mu2_err = gauss_fit(
            fit_script,
            data_dir / fname
        )
        # collect results from this fit
        angles.append(deg)
        theta_err.append(sigma_theta)
        mu1_list.append(mu1)
        mu2_list.append(mu2)
        mu1_err_list.append(mu1_err)
        mu2_err_list.append(mu2_err)

    angles = np.array(angles)
    theta_err = np.array(theta_err)
    mu1 = np.array(mu1_list)
    mu2 = np.array(mu2_list)
    mu1_err = np.array(mu1_err_list)
    mu2_err = np.array(mu2_err_list)

    # ----------------------------
    # Effective uncertainties
    # ----------------------------
    sigma_mu1_eff = np.sqrt(
        mu1_err**2 +
        (dcompton_dtheta(angles, E1, ME_C2) * theta_err)**2
    )

    sigma_mu2_eff = np.sqrt(
        mu2_err**2 +
        (dcompton_dtheta(angles, E2, ME_C2) * theta_err)**2
    )


    # ----------------------------
    # Fit Compton curves
    # ----------------------------

    popt1, pcov1 = curve_fit(
        lambda th, me: compton_energy(th, E1, me),
        angles,
        mu1,
        sigma=sigma_mu1_eff,
        absolute_sigma=True,
        p0=[ME_C2]
    )

    popt2, pcov2 = curve_fit(
        lambda th, me: compton_energy(th, E2, me),
        angles,
        mu2,
        sigma=sigma_mu2_eff,
        absolute_sigma=True,
        p0=[ME_C2]
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

    '''
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
    '''
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
