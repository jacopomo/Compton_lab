#!/usr/bin/env python3

import sys
import numpy as np
import subprocess
import re
from pathlib import Path
import matplotlib.pyplot as plt
import json
from scipy import stats

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
    # Residuals (normalized) and chi2
    # ----------------------------
    # model predictions using real ME value
    model1 = compton_energy(angles, E1, ME_C2)
    model2 = compton_energy(angles, E2, ME_C2)

    # normalized residuals (using the variances above)
    res1 = (mu1 - model1) / sigma_mu1_eff
    res2 = (mu2 - model2) / sigma_mu2_eff

    # chi2 and ndof using total least-squares variance
    chi2_1 = np.sum(res1**2)
    chi2_2 = np.sum(res2**2)
    ndof_1 = len(angles) - 1
    ndof_2 = len(angles) - 1

    # ----------------------------
    # Plot with fits and error bars
    # ----------------------------
    th_plot = np.linspace(angles.min(), angles.max(), 500)

    # create main + residuals subplot
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(7, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # upper: data and model (original errorbars kept)
    ax1.errorbar(angles, mu1, xerr=theta_err, yerr=mu1_err, label="μ₁ (1173.2 keV)", fmt="o", capsize=3, color="C0")
    ax1.errorbar(angles, mu2, xerr=theta_err, yerr=mu2_err, label="μ₂ (1332.5 keV)", fmt="s", capsize=3, color="C1")

    # plot reference Compton curves at 511 keV
    ax1.plot(th_plot, compton_energy(th_plot, E1, 511.0), ":", label="Modello 1173.2 keV", color="C0", alpha=0.7)
    ax1.plot(th_plot, compton_energy(th_plot, E2, 511.0), ":", label="Modello 1332.5 keV", color="C1", alpha=0.7)

    ax1.set_ylabel("Posizione dei picchi μ [keV]")
    ax1.set_title("Verifica scattering Compton")
    ax1.legend()
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)


    # Configuration for both lines
    text_props = dict(
        transform=ax1.transAxes,
        va="bottom",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )

    # First line (C0)
    ax1.text(
        0.02, 0.09, 
        f"$\\chi^2_{{\\mathrm{{eff}}}}$/ndof (1173.2 keV) = {chi2_1:.1f}/{ndof_1} = {chi2_1/ndof_1:.2f}",
        color="C0", 
        **text_props
    )

    # Second line (C1) - shifted down slightly (adjust 0.05 based on your font size/scaling)
    ax1.text(
        0.02, 0.02, 
        f"$\\chi^2_{{\\mathrm{{eff}}}}$/ndof (1332.5 keV) = {chi2_2:.1f}/{ndof_2} = {chi2_2/ndof_2:.2f}",
        color="C1", 
        **text_props
    )

    # lower: residuals plot (normalized)
    ax2.axhline(0.0, color="k", ls="--", alpha=0.7)
    ax2.plot(angles, res1, "o", color="C0", label="resid μ₁")
    ax2.plot(angles, res2, "s", color="C1", label="resid μ₂")
    ax2.set_xlabel("Angolo di scattering [deg]")
    ax2.set_ylabel("Residui efficaci normalizzati")
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.show()

    p_value = stats.chi2.sf(chi2_1, ndof_1)
    print(f"P-value for 1173.2 keV: {p_value:.4e}")

    p_value = stats.chi2.sf(chi2_2, ndof_2)
    print(f"P-value for 1332.5 keV: {p_value:.4e}")

if __name__ == "__main__":
    main()
