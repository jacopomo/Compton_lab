#!/usr/bin/env python3

from html import parser
import numpy as np
import argparse
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import chi2
import matplotlib.pyplot as plt
import re
import json
import matplotlib.gridspec as gridspec


plt.rcParams.update({'font.size': 20})

def compton_peak(E_gamma, theta_deg):
    theta = np.deg2rad(theta_deg)
    mec2 = 511.0  # keV
    Eprime = E_gamma / (1 + (E_gamma / mec2) * (1 - np.cos(theta)))
    return Eprime


def double_gauss(E, mu1, mu2, sigma1, sigma2, A1, A2):
    return (
        A1 * np.exp(-(E - mu1)**2 / (2 * sigma1**2)) +
        A2 * np.exp(-(E - mu2)**2 / (2 * sigma2**2))
    )


def triple_gauss(E, mu1, mu2, mu3, sigma1, sigma2, sigma3, A1, A2, A3):
    return (
        A1 * np.exp(-(E - mu1)**2 / (2 * sigma1**2)) +
        A2 * np.exp(-(E - mu2)**2 / (2 * sigma2**2)) +
        A3 * np.exp(-(E - mu3)**2 / (2 * sigma3**2))
    )

def neg2loglike(params, E, data, use_triple, E_cutoff):
    if use_triple:
        mu1, mu2, mu3, s1, s2, s3, A1, A2, A3 = params
        model = triple_gauss(E, mu1, mu2, mu3, s1, s2, s3, A1, A2, A3)
    else:
        mu1, mu2, s1, s2, A1, A2 = params
        model = double_gauss(E, mu1, mu2, s1, s2, A1, A2)

    model = np.clip(model, 1e-12, None)
    mask = (data > 0) & (E <= E_cutoff)

    nll = 2 * (
        np.sum(model - data) +
        np.sum(data[mask] * np.log(data[mask] / model[mask]))
    )

    # resolution priors
    nll += ((s1 - np.sqrt(mu1)) / (0.5 * np.sqrt(mu1)))**2
    nll += ((s2 - np.sqrt(mu2)) / (0.5 * np.sqrt(mu2)))**2

    return nll

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Angle (deg) or full filename")
    parser.add_argument(
        "-v", "--visualize",
        action="store_true",
        help="Show fit plot"
    )
    args = parser.parse_args()
    
    root = Path(__file__).resolve().parents[1]

    config_path = root / "Analisi" / "fit_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    # ----------------------------
    # Load data
    # ----------------------------
    data_dir = (
        root
        / "Dati"
        / "Measures"
        / "Angles"
        / "Calibrati"
    )

    arg = args.input
    if arg.endswith(".txt"):
        filepath = Path(arg)

        # extract angle from filename, e.g. "15deg_251125_EnergieC.txt"
        m = re.match(r"(\d+)deg_", filepath.name)
        if m is None:
            raise ValueError(
                f"Cannot extract angle from filename '{filepath.name}'"
            )
        deg = int(m.group(1))
    else:
        deg = int(arg)
        files = list(data_dir.glob(f"{deg}deg_*_EnergieC.txt"))
        if len(files) != 1:
            raise RuntimeError(
                f"Expected exactly one file for {deg} deg, found {len(files)}"
            )
        filepath = files[0]

    energy, counts_raw = np.loadtxt(filepath, unpack=True, skiprows=1)

    cfg = config.get(filepath.name, {})
    use_triple = cfg.get("model", "double") == "triple"
    init = cfg.get("init", {})
    bins = cfg.get("bins", 100)


    edges = np.linspace(energy.min(), energy.max(), bins + 1)
    counts, energy_bins = np.histogram(energy, bins=edges, weights=counts_raw)
    counts_copy, _ = np.histogram(energy, bins=edges, weights=counts_raw)
    energy = 0.5 * (edges[:-1] + edges[1:])

    # physical cut
    E_cutoff = init.get("E_cutoff", 1500.0)  # keV
    print(f"Using E_cutoff = {E_cutoff} keV")
    counts[energy > E_cutoff] = 0.0

    mu1_0 = init.get("mu1", 0.95 * compton_peak(1332.0, deg))
    mu2_0 = init.get("mu2", 0.95 * compton_peak(1173.0, deg))

    s1_0 = init.get("sigma1", np.sqrt(mu1_0))
    s2_0 = init.get("sigma2", np.sqrt(mu2_0))

    A1_0 = init.get("A1", 0.6 * np.max(counts))
    A2_0 = init.get("A2", 0.9 * A1_0)

    if use_triple:
        mu3_0 = init.get("mu3", 0.8 * mu1_0)
        s3_0 = init.get("sigma3", np.sqrt(mu3_0))
        A3_0 = init.get("A3", 0.5 * A1_0)

        x0 = [mu1_0, mu2_0, mu3_0, s1_0, s2_0, s3_0, A1_0, A2_0, A3_0]
        bounds = [
            ((mu1_0+mu2_0)/2, E_cutoff),
            ((mu3_0+mu2_0)/2,(mu1_0+mu2_0)/2),
            (energy.min(), (mu3_0+mu2_0)/2),
            (3*np.max(energy)/bins, None),
            (3*np.max(energy)/bins, None),
            (3*np.max(energy)/bins, None),
            (1e-6, 2 * np.max(counts)),
            (1e-6, 2 * np.max(counts)),
            (1e-6, 2 * np.max(counts)),
        ]
    else:
        x0 = [mu1_0, mu2_0, s1_0, s2_0, A1_0, A2_0]
        bounds = [
            (energy.min(), E_cutoff),
            (energy.min(), mu1_0),
            (3*np.max(energy)/bins, None),
            (3*np.max(energy)/bins, None),
            (1e-6, 2 * np.max(counts)),
            (1e-6, 2 * np.max(counts)),
        ]


    # ----------------------------
    # Fit
    # ----------------------------
    res = minimize(
        neg2loglike,
        x0=x0,
        args=(energy, counts, use_triple, E_cutoff),
        method="L-BFGS-B",
        bounds=bounds,
    )

    # ----------------------------
    # Results
    # ----------------------------
    print("Converged:", res.success)
    print("Message:", res.message)
    print()
    names = (
        ["mu1", "mu2", "mu3", "sigma1", "sigma2", "sigma3", "A1", "A2", "A3"]
        if use_triple
        else ["mu1", "mu2", "sigma1", "sigma2", "A1", "A2"]
    )  
    cov = res.hess_inv.todense()
    errs = np.sqrt(np.diag(cov))


    for n, v, e, b in zip(names, res.x, errs, bounds):
            print(f"{n:6s} = {v:.3f} ± {e:.3f}  ({b})")


    ndof = np.count_nonzero(counts > 0) - len(res.x)
    pval = 1 - chi2.cdf(res.fun, ndof)

    print(f"\n-2 ln L = {res.fun:.2f}")
    print(f"ndof    = {ndof}")
    print(f"p-value = {pval:.3e}")

    # ----------------------------
    # Visualization
    # ----------------------------
    if args.visualize:
        Eplot = np.linspace(energy.min(), energy.max(), 2000)

        if use_triple:
            mu1, mu2, mu3, s1, s2, s3, A1, A2, A3 = res.x

            g1 = A1 * np.exp(-(Eplot - mu1)**2 / (2 * s1**2))
            g2 = A2 * np.exp(-(Eplot - mu2)**2 / (2 * s2**2))
            g3 = A3 * np.exp(-(Eplot - mu3)**2 / (2 * s3**2))

            model_plot = g1 + g2 + g3

            # model evaluated at data points (for residuals)
            model_data = (
                A1 * np.exp(-(energy - mu1)**2 / (2 * s1**2)) +
                A2 * np.exp(-(energy - mu2)**2 / (2 * s2**2)) +
                A3 * np.exp(-(energy - mu3)**2 / (2 * s3**2))
            )

        else:
            mu1, mu2, s1, s2, A1, A2 = res.x

            g1 = A1 * np.exp(-(Eplot - mu1)**2 / (2 * s1**2))
            g2 = A2 * np.exp(-(Eplot - mu2)**2 / (2 * s2**2))

            model_plot = g1 + g2

            model_data = (
                A1 * np.exp(-(energy - mu1)**2 / (2 * s1**2)) +
                A2 * np.exp(-(energy - mu2)**2 / (2 * s2**2))
            )

        # --- Sigma over angle ---    
        theta_error = cfg.get("theta_error", 1.0)
        N1 = A1 * np.sqrt(2 * np.pi) * s1
        N2 = A2 * np.sqrt(2 * np.pi) * s2

        sigma_theta = theta_error / np.sqrt((N1+N2))

        print(f"sigma_theta = {sigma_theta:.2e}")

        # --- Residuals (normalized) ---
        sigma = np.sqrt(np.maximum(counts, 1))  # avoid division by zero
        residuals = (counts - model_data) / sigma

        # --- Plot ---
        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(7, 6),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True
        )

        # Main plot
        ax.stairs(
            counts_copy, energy_bins,
            color="gray"
        )

        ax.errorbar(
            energy[counts != 0], counts[counts != 0],
            yerr=np.sqrt(counts[counts != 0]),
            fmt='.',
            label="Dati", color="blue"
        )

        ax.plot(Eplot, model_plot, '-', color="red", lw=2.5, label="Fit totale")
        ax.plot(Eplot, g1, 'r--', lw=2, label="Gaussiana 1")
        ax.plot(Eplot, g2, 'r:',  lw=2, label="Gaussiana 2")

        if use_triple:
            ax.plot(Eplot, g3, 'b--', lw=1.5, label="Spalla Compton")

        ax.set_ylabel("Conteggi")
        ax.set_title(f"{'Tripla' if use_triple else 'Doppia'} Gaussiana ({deg}°)")
        ax.legend()

        # Residuals
        axr.axhline(0, color='black', lw=1)
        axr.errorbar(
            energy[counts != 0], residuals[counts != 0],
            fmt='.', color="blue"
        )

        axr.set_xlabel("Energy [keV]")
        axr.set_ylabel("Residui")

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    main()
