#!/usr/bin/env python3

import sys
import numpy as np
import subprocess
import re
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import h5py
from scipy.special import erf
from scipy.optimize import minimize


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

def gauss_cdf(x, mu, sigma):
    return 0.5 * (1 + erf((x - mu) / (np.sqrt(2) * sigma)))

def load_h5(deg):
    root = Path(__file__).resolve().parents[1]

    h5file = (
        root
        / "Montecarlo"
        / "results"
        / "2d_histograms"
        / f"{deg}deg.h5"
    )

    with h5py.File(h5file, "r") as f:
        H = f["H"][:]
        xedges = f["xedges"][:]
        yedges = f["yedges"][:]

    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])

    return H, xedges, yedges, xcenters, ycenters

def model_projection(params, H, xcenters, ycenters):
    mu_x, sig_x, mu_y, sig_y = params

    fx = gauss_cdf(xcenters, mu_x, sig_x)[:, None] 
    fy = gauss_cdf(ycenters, mu_y, sig_y)[None, :] 

    Hf = (H * fx * fy).sum(axis=1) # sum over plastic (y)
    Hfn = Hf/(Hf.max()) # normalized histogram
    Hfin = Hfn
    return Hfin

def gauss_fit(deg, energy, counts, triple):
    # ----------------------------
    # Physical energy cut (Co-60 max line)
    # ----------------------------
    E_MAX = 1500.0  # keV

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

    if triple:
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
    def neg2loglike(params, E, data, triple):
        if triple:
            mu1, mu2, mu3, s1, s2, s3, A1, A2, A3 = params
            model = triple_gauss(E, mu1, mu2, mu3, s1, s2, s3, A1, A2, A3)
        else:
            mu1, mu2, s1, s2, A1, A2 = params
            model = double_gauss(E, mu1, mu2, s1, s2, A1, A2)

        model = np.clip(model, 1e-12, None)
        mask = (data > 0) & (E <= 1500.0)

        nll = 2 * (
            np.sum(model - data) +
            np.sum(data[mask] * np.log(data[mask] / model[mask]))
        )

    res = minimize(
        neg2loglike,
        x0=x0,
        args=(energy, counts, triple),
        method="L-BFGS-B",
        bounds=bounds
    )

    # ----------------------------
    # Results
    # ----------------------------
    print("\n=== FIT RESULTS ===")
    print("Converged:", res.success)
    print("Message:", res.message)
    print()

    if triple:
        names = ["mu1", "mu2", "mu3", "sigma1", "sigma2", "sigma3", "A1", "A2", "A3"]
    else:
        names = ["mu1", "mu2", "sigma1", "sigma2", "A1", "A2"]   
    cov = res.hess_inv.todense()
    errs = np.sqrt(np.diag(cov))

    for n, v, e in zip(names, res.x, errs):
        print(f"{n:6s} = {v:.3f} ± {e:.3f}")

    mu1, mu2 = res.x[:2]


    return mu1, mu2

def main():
    # ----------------------------
    # DEFS
    # ----------------------------

    deg = 15

    H, xedges, yedges, xcenters, ycenters = load_h5(deg)

    H_no_plastic = model_projection([750, 30, 0, 1e-6], H, xcenters, ycenters)
    gauss_fit(deg, xcenters, H_no_plastic, True)

if __name__ == "__main__":
    main()








