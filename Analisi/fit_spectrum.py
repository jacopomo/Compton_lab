#!/usr/bin/env python3

from matplotlib.colors import ListedColormap
import numpy as np
import h5py
import argparse
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import erf
from scipy.stats import chi2
import matplotlib.pyplot as plt
import json
from scipy.stats import norm

def load_resolution_coeffs(root, date_str):
    """
    date_str: '021225' -> looks for '02_12_25'
    """
    key = f"{date_str[:2]}_{date_str[2:4]}_{date_str[4:]}"
    calib_file = (
        root
        / "Dati"
        / "Calibration"
        / "Processed"
        / "calibration_table.json"
    )

    with open(calib_file, "r") as f:
        calib = json.load(f)

    if key not in calib:
        raise KeyError(f"No calibration for date {key}")

    return calib[key]["coeff risol"]  # a, b, c


def smear_crystal_axis(H, xcenters, a, b, c):
    """
    Smear MC histogram along x (crystal energy) using
    sigma^2 = a E^2 + b E + c
    """
    nx, ny = H.shape
    Hsmeared = np.zeros_like(H)

    dx = np.diff(xcenters).mean()

    for i, E in enumerate(xcenters):
        sigma2 = a * E**2 + b * E + c
        if sigma2 <= 0:
            Hsmeared[i] += H[i]
            continue

        sigma = np.sqrt(sigma2)

        # Gaussian weights over x bins
        weights = norm.pdf(xcenters, loc=E, scale=sigma)
        weights /= weights.sum()  # conserve counts

        # Distribute counts
        Hsmeared += weights[:, None] * H[i][None, :]

    return Hsmeared

# ----------------------------
# Filters
# ----------------------------
def gauss_cdf(x, mu, sigma):
    return 0.5 * (1 + erf((x - mu) / (np.sqrt(2) * sigma)))

# ----------------------------
# Forward model
# ----------------------------
def model_projection(params, H, xcenters, ycenters):
    mu_x, sig_x, mu_y, sig_y = params

    B = 0

    fx = gauss_cdf(xcenters, mu_x, sig_x)[:, None] 
    fy = ((1-B)*gauss_cdf(ycenters, mu_y, sig_y) + B)[None, :] 

    Hf = (H * fx * fy).sum(axis=1) # sum over plastic (y)
    Hfn = Hf/(Hf.max()) # normalized histogram
    Hfin = A*Hfn
    return Hfin


# ----------------------------
# Poisson -2 log L
# ----------------------------
def neg2loglike(params, H, xcenters, ycenters, data):
    model = model_projection(params, H, xcenters, ycenters)
    model = np.clip(model, 1e-12, None)
    data  = np.clip(data,  0.0, None)

    mask = data > 0
    return 2.0 * (
        np.sum(model - data) +
        np.sum(data[mask] * np.log(data[mask] / model[mask]))
    )
# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("deg", type=int, help="Detector angle in degrees (e.g. 20)")
    parser.add_argument(
    "-v", "--visualize",
    action="store_true",
    help="Show sanity-check plots"
)
    parser.add_argument(
    "-b", "--bins",
    type=int,
    default=1000,
    help="Number of bins (default: 60)"
)
    parser.add_argument(
    "-d", "--debug",
    action='store_true',
    default=False,
    help="Activate debug histograms."
)

    args = parser.parse_args()

    deg = args.deg
    vis = args.visualize
    debug = args.debug
    n_bins = args.bins
    root = Path(__file__).resolve().parents[1]

    # ----------------------------
    # Load MC histogram
    # ----------------------------
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
    
    # Rebin MC 2D histogram to desired number of bins
    xmin, xmax = xedges[0], xedges[-1]
    xedges_new = np.linspace(xmin, xmax, n_bins + 1)

    indices = np.searchsorted(xedges, xedges_new, side="left")
    indices = np.clip(indices, 0, len(xedges) - 1)

    H = np.add.reduceat(H, indices[:-1], axis=0)
    xedges = xedges_new
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])

    xedges = xedges_new
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    data_dir = (
        root
        / "Dati"
        / "Measures"
        / "Angles"
        / "Calibrati"
    )

    files = list(data_dir.glob(f"{deg}deg_*_EnergieC.txt"))
    if len(files) == 0:
        raise FileNotFoundError("No data file found")

    data_file = files[0]
    date_str = data_file.stem.split("_")[1]  # '021225'

    a, b, c = load_resolution_coeffs(root, date_str)

    data_energy, data_counts = np.loadtxt(files[0], unpack=True, skiprows=1)

    # Bin data onto new MC x-binning
    data_binned, _ = np.histogram(data_energy, bins=xedges, weights=data_counts)
    data_binned = data_binned/data_binned.sum()
    
    # ----------------------------
    # Apply crystal energy smearing
    # ----------------------------
    H_raw = H.copy()
    H = smear_crystal_axis(H, xcenters, a, b, c)
    #H = H_raw

    # ----------------------------
    # Fit
    # ----------------------------
    A0 = max(data_binned)
    
    x0 = [
        700.0,      # mu_x
        25,         # sig_x
        90,       # mu_y
        20,        # sig_y
        A0           # Maximum peak high
    ]

    bounds = [
        (650, 800),
        (10, 30),
        (80, 100),
        (5, 70),
        (A0 - 5*np.sqrt(A0), A0 + 5*np.sqrt(A0))
    ]

    res = minimize(
        neg2loglike,
        x0=x0,
        args=(H, xcenters, ycenters, data_binned),
        method="L-BFGS-B",
        bounds=bounds,
    )

    success = res.success
    print(".............................................")
    print(f"Succes Fut: {success}")
    print("'''''''''''''''''''''''''''''''''''''''''''''")
    best_params = res.x

    # Best-fit filtered histograms
    mu_x, sig_x, mu_y, sig_y, A= best_params

    #H_proj = model_projection(best_params, H, xcenters, ycenters)
    H_proj = model_projection([735, 25, 275, 40, 7000], H, xcenters, ycenters)

    fx = gauss_cdf(xcenters, mu_x, sig_x)[:, None] 
    fy = gauss_cdf(ycenters, mu_y, sig_y)[None, :]
    H_filtered = (H * fx * fy)

    H_raw_proj = model_projection(best_params, H_raw, xcenters, ycenters)

    # ----------------------------
    # Results
    # ----------------------------
    print("\n=== FIT RESULTS ===")
    print("Converged:", res.success)
    print("Message:", res.message)
    print()

    names = ["mu_x", "sigma_x", "mu_y", "sigma_y"]
    cov = res.hess_inv.todense()
    errs = np.sqrt(np.diag(cov))

    for n, v, e, b in zip(names, res.x, errs, bounds):
        print(f"{n:8s} = {v:.4g} ± {e:.4g}  ({e/v*100:.2g} % relative error) [{b}]")

    # ----------------------------
    # Goodness of fit
    # ----------------------------
    mask = data_binned > 0
    ndof = np.count_nonzero(mask) - len(res.x)
    chi2_val = res.fun
    pval = 1.0 - chi2.cdf(chi2_val, ndof)

    print("\n=== GOODNESS OF FIT ===")
    print(f"-2 ln L = {chi2_val:.2f}")
    print(f"ndof    = {ndof}")
    print(f"p-value = {pval:.3f}")

    print(f"\ncov =\n")
    for j in range(len(cov[:, 0])):
        print([f"{x/(errs[i] * errs[j]):.2f}  " for i,x in enumerate(cov[j, :])])


    # ----------------------------
    # Visualization
    # ----------------------------
    def transparent_zero_cmap(base="viridis"):
        cmap = plt.get_cmap(base)
        colors = cmap(np.linspace(0, 1, 256))
        colors[0] = [1, 1, 1, 0]
        return ListedColormap(colors)
    if debug:
        
        # --- 1. Original MC 2D histogram ---
        newcmap = transparent_zero_cmap()
        plt.figure(figsize=(7, 5))
        plt.pcolormesh(
            xedges, yedges, H_raw.T,
            cmap=newcmap,
            shading="auto"
        )
        plt.xlabel("Energy in crystal")
        plt.ylabel("Energy in plastic")
        plt.title(f"Original MC 2D histogram ({deg} deg)")
        plt.colorbar(label="Counts")
        plt.tight_layout()
        plt.show()
        # --- 2. Smeared MC 2D histogram ---
        newcmap = transparent_zero_cmap()
        plt.figure(figsize=(7, 5))
        plt.pcolormesh(
            xedges, yedges, H.T,
            cmap=newcmap,
            shading="auto"
        )
        plt.xlabel("Energy in crystal")
        plt.ylabel("Energy in plastic")
        plt.title(f"Original MC 2D histogram ({deg} deg)")
        plt.colorbar(label="Counts")
        plt.tight_layout()
        plt.show()
        # --- 3. Filtered MC 2D histogram ---
        newcmap = transparent_zero_cmap()
        plt.figure(figsize=(7, 5))
        plt.pcolormesh(
            xedges, yedges, H_filtered.T,
            cmap=newcmap,
            shading="auto"
        )
        plt.xlabel("Energy in crystal")
        plt.ylabel("Energy in plastic")
        plt.title(f"Filtered MC 2D histogram (best fit {deg} deg)")
        plt.colorbar(label="Counts")
        plt.tight_layout()
        plt.show()

    if debug or vis:
        # --- 4. 1D projection vs data ---
        
        plt.figure(figsize=(7, 5))
        plt.step(
            xcenters,
            data_binned,
            where="mid",
            label="Data",
            linewidth=1.5
        )
        plt.step(
            xcenters,
            H_proj,
            where="mid",
            label="MC (filtered)",
            linewidth=2.5
        )
        #plt.step(
        #    xcenters,
        #    H_raw_proj,
        #    where="mid",
        #    label="MC_raw (filtered)",
        #    linewidth=1,
        #)
        plt.xlabel("Energy in crystal")
        plt.ylabel("Probability")
        plt.title(f"Crystal energy spectrum ({deg} deg)")
        plt.legend()
        plt.tight_layout()

        plt.show()

if __name__ == "__main__":
    main()
