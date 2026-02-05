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

plt.rcParams.update({'font.size': 20})

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
def gauss_fit(script, filepath, vis=False):
    cmd = [
        sys.executable,
        str(script),
        filepath,
    ]
    if vis:
        cmd = cmd + ["-v"]

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
        #sigma_theta = cfg["theta_error"]
        sigma_theta = 0.7
        use_triple = (cfg.get("model", "double") == "triple")

        print(f"  {deg:>3} deg | triple={use_triple} | {fname}")

        mu1, mu2, mu1_err, mu2_err = gauss_fit(
            fit_script,
            data_dir / fname,
            False
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

    # -----------------------------
    # Test del Chi^2
    # -----------------------------

    # Calcolo dei residui per il secondo subplot
    residuals_mu1 = mu1 - compton_energy(angles, E1, 511.0)
    residuals_mu2 = mu2 - compton_energy(angles, E2, 511.0)

    sigma_mu1_eff = np.sqrt(
        mu1_err**2 +
        (dcompton_dtheta(angles, E1, ME_C2) * theta_err)**2
    )

    sigma_mu2_eff = np.sqrt(
        mu2_err**2 +
        (dcompton_dtheta(angles, E2, ME_C2) * theta_err)**2
    )

    chi1 = np.sum((residuals_mu1/sigma_mu1_eff)**2)
    chi2 = np.sum((residuals_mu2/sigma_mu2_eff)**2)

    print("###################################")
    print("#####  TEST DEL CHI^2   ###########")
    print("###################################")

    print(f"chi2_mu1 = {chi1}")
    print(f"chi2_mu2 = {chi2}")

    # ----------------------------
    # Plot with fits and error bars
    # ----------------------------
    # Esegui il linspace per i valori di angolo
    th_plot = np.linspace(angles.min(), angles.max(), 500)

    # Creiamo due subplots: uno sopra (per i dati) e uno sotto (per i residui)
    fig, axs = plt.subplots(2, 1, figsize=(7, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Plot dei dati nel primo subplot
    axs[0].errorbar(angles, mu1, xerr=theta_err, yerr=mu1_err, label="μ₁ (1332.5 keV)", fmt="o", markersize = 10, linewidth=5, capsize=5, capthick=2, color="C0")
    axs[0].errorbar(angles, mu2, xerr=theta_err, yerr=mu2_err, label="μ₂ (1173.2 keV)", fmt="o", markersize = 10, capsize=5, linewidth=5, capthick=2, color="C1")

    axs[0].text(5, 1280, f"χ² / ndof: {chi1/8:.1f}", color="midnightblue")
    axs[0].text(3, 1150, f"χ² / ndof: {chi2/8:.1f}", color="sienna")

    # Plot delle curve teoriche nel primo subplot
    axs[0].plot(
        th_plot,
        compton_energy(th_plot, E1, 511.0),
        ":",
        label="Modello 1173.2 keV",
        color="black",
        alpha=0.5,
        linewidth=4
    )
    axs[0].plot(
        th_plot,
        compton_energy(th_plot, E2, 511.0),
        ":",
        label="Modello 1332.5 keV",
        color="red",
        alpha=0.5,
        linewidth=4
    )

    # Impostazioni del primo subplot
    axs[0].set_ylabel("Posizione dei picchi [keV]")
    axs[0].set_title("Compton scattering verification")
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # Plot dei residui nel secondo subplot
    axs[1].errorbar(angles, residuals_mu1, yerr=mu1_err, fmt='o', markersize = 10, linewidth=5, capsize=5, capthick=2, label="Residuals μ₁", color="C0")
    axs[1].errorbar(angles, residuals_mu2, yerr=mu2_err, fmt='o', markersize = 10, linewidth=5, capsize=5, capthick=2, label="Residuals μ₂", color="C1")
    axs[1].axhline(0, color='gray', linewidth=1.5, linestyle='--')

    # Impostazioni del secondo subplot
    axs[1].set_xlabel("Angolo di scattering θ [deg]")
    axs[1].set_ylabel("Residui [keV]")
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # Layout
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
