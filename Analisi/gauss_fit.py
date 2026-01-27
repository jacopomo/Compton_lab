#!/usr/bin/env python3

from html import parser
import numpy as np
import argparse
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import chi2
import matplotlib.pyplot as plt
import re

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

def neg2loglike(params, E, data, triple):
    if triple:
        mu1, mu2, mu3, s1, s2, s3, A1, A2, A3 = params
        model = triple_gauss(E, mu1, mu2, mu3, s1, s2, s3, A1, A2, A3)
    else:
        mu1, mu2, s1, s2, A1, A2 = params
        model = double_gauss(E, mu1, mu2, s1, s2, A1, A2)

    model = np.clip(model, 1e-12, None)
    mask = (data > 0) & (E <= 1330.0)

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
        "-b", "--bins",
        type=int,
        default=100,
        help="Rebin data to this many bins"
    )
    parser.add_argument(
        "-v", "--visualize",
        action="store_true",
        help="Show fit plot"
    )
    parser.add_argument(
    "-t", "--triple",
    action="store_true",
    help="Use triple Gaussian (adds Compton shoulder)"
)

    args = parser.parse_args()
    
    root = Path(__file__).resolve().parents[1]

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

    energy, counts = np.loadtxt(filepath, unpack=True, skiprows=1)

    # ----------------------------
    # Optional rebinning
    # ----------------------------
    if args.bins is not None:
        edges = np.linspace(energy.min(), energy.max(), args.bins + 1)
        counts, _ = np.histogram(energy, bins=edges, weights=counts)
        energy = 0.5 * (edges[:-1] + edges[1:])

    # ----------------------------
    # Physical energy cut (Co-60 max line)
    # ----------------------------
    E_MAX = 1330.0  # keV

    cut_mask = energy <= E_MAX
    counts = counts.copy()
    counts[~cut_mask] = 0.0

    mu1_0 = 0.95 * compton_peak(1173.0, deg)
    mu2_0 = 0.95 * compton_peak(1330.0, deg)
    if mu1_0 > mu2_0:
        mu1_0, mu2_0 = mu2_0, mu1_0

    s1_0 = np.sqrt(mu1_0)
    s2_0 = np.sqrt(mu2_0)

    A1_0 = 0.6 * np.max(counts)
    A2_0 = 0.9 * A1_0

    if args.triple:
        mu3_0 = 0.8 * mu1_0
        s3_0 = np.sqrt(mu3_0)
        A3_0 = 0.5 * A1_0

        x0 = [mu1_0, mu2_0, mu3_0, s1_0, s2_0, s3_0, A1_0, A2_0, A3_0]
        bounds = [
            (energy.min(), energy.max()),
            (mu1_0, energy.max()),
            (energy.min(), mu1_0),
            (1e-3, None),
            (1e-3, None),
            (1e-3, None),
            (1e-6, None),
            (1e-6, None),
            (1e-6, None),
        ]
    else:
        x0 = [mu1_0, mu2_0, s1_0, s2_0, A1_0, A2_0]
        bounds = [
            (energy.min(), energy.max()),
            (mu1_0, energy.max()),
            (1e-3, None),
            (1e-3, None),
            (1e-6, None),
            (1e-6, None),
        ]


    # ----------------------------
    # Fit
    # ----------------------------
    res = minimize(
        neg2loglike,
        x0=x0,
        args=(energy, counts, args.triple),
        method="L-BFGS-B",
        bounds=bounds
    )

    # ----------------------------
    # Results
    # ----------------------------
    print("\n=== DOUBLE GAUSSIAN FIT (6 params) ===")
    print("Converged:", res.success)
    print("Message:", res.message)
    print()

    if args.triple:
        names = ["mu1", "mu2", "mu3", "sigma1", "sigma2", "sigma3", "A1", "A2", "A3"]
    else:
        names = ["mu1", "mu2", "sigma1", "sigma2", "A1", "A2"]   
    cov = res.hess_inv.todense()
    errs = np.sqrt(np.diag(cov))

    for n, v, e in zip(names, res.x, errs):
        print(f"{n:6s} = {v:.3f} ± {e:.3f}")

    # ----------------------------
    # Goodness of fit
    # ----------------------------
    mask = counts > 0
    ndof = np.count_nonzero(mask) - len(res.x)
    chi2_val = res.fun
    pval = 1.0 - chi2.cdf(chi2_val, ndof)

    print("\n=== GOODNESS OF FIT ===")
    print(f"-2 ln L = {chi2_val:.2f}")
    print(f"ndof    = {ndof}")
    print(f"p-value = {pval:.7f}")

    # ----------------------------
    # Visualization
    # ----------------------------
    if args.visualize:
        import matplotlib.gridspec as gridspec

        Eplot = np.linspace(energy.min(), energy.max(), 2000)
        if args.triple:
            mu1, mu2, mu3, sigma1, sigma2, sigma3, A1, A2, A3 = res.x
        else:
            mu1, mu2, sigma1, sigma2, A1, A2 = res.x

        if args.triple:
            model = triple_gauss(Eplot, *res.x)
        else:
            model = double_gauss(Eplot, *res.x)        
        g1 = A1 * np.exp(-(Eplot - mu1)**2 / (2 * sigma1**2))
        g2 = A2 * np.exp(-(Eplot - mu2)**2 / (2 * sigma2**2))
        if args.triple:
            g3 = A3 * np.exp(-(Eplot - mu3)**2 / (2 * sigma3**2))

        # Poisson errors
        yerr = np.sqrt(counts)
        yerr[yerr == 0] = 1.0  # avoid zero error bars

        # Residuals: (data - model) / sqrt(data)
        if args.triple:
            model_at_bins = triple_gauss(energy, mu1, mu2, mu3, sigma1, sigma2, sigma3, A1, A2, A3)
        else:
            model_at_bins = double_gauss(energy, mu1, mu2, sigma1, sigma2, A1, A2)
        residuals = (counts - model_at_bins) / np.where(counts > 0, np.sqrt(counts), 1.0)

        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

        # Top: data + fit
        ax0 = fig.add_subplot(gs[0])
        ax0.errorbar(energy, counts, yerr=yerr, fmt='o', label="Data", color='black', alpha=0.7, markersize=4)
        ax0.plot(Eplot, model, 'r-', label="Total fit", linewidth=2)
        ax0.plot(Eplot, g1, 'r--', alpha=0.7, label="Gaussian 1")
        ax0.plot(Eplot, g2, 'r:', alpha=0.7, label="Gaussian 2")
        if args.triple:
            ax0.plot(Eplot, g3, 'b-.', alpha=0.7, label="Spalla Compton")
        ax0.set_ylabel("Counts")
        ax0.legend()
        if args.triple:
            plt.title(f"Triple Gaussian fit ({deg} deg)")
        else:
            plt.title(f"Double Gaussian fit ({deg} deg)")
        ax0.set_xticklabels([])  # hide x labels, for cleaner residual panel

        # Bottom: residuals
        ax1 = fig.add_subplot(gs[1])
        ax1.errorbar(energy, residuals, yerr=np.ones_like(residuals), fmt='o', color='black', alpha=0.7, markersize=4)
        ax1.axhline(0, color='black', linestyle='--', alpha=0.7)
        ax1.set_xlabel("Energy in crystal [keV]")
        ax1.set_ylabel("Residuals")
        ax1.set_ylim(-5, 5)  # adjust for visibility

        plt.subplots_adjust(hspace=0.05, left=0.1, right=0.95, top=0.95, bottom=0.1)
        plt.show()

if __name__ == "__main__":
    main()
