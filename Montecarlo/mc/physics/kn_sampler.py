# mc/physics/kn_sampler.py

import numpy as np
import scipy.integrate as integrate

from mc.physics.compton import compton
from mc.config import RE, E1, E2, E_GRID, MU_GRID
from mc.utils.math3d import to_full_array

def kn(E, mu):
    """Klein-Nishima differential cross section formula

    Args:
        E (nparray): (n,) initial photon energies [keV]
        mu (nparray): (n,) cosine of scattering angles [rads]

    Returns:
        nparray: (n,) differential cross section [cm^2]
    """
    lr = compton(E, mu) / E
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 0.5 * (RE**2) * (lr**2) * (lr + 1.0/lr - 1 + mu**2)
    out = np.where(np.isfinite(out), out, 0.0)
    return np.clip(out, 0.0, None)  

def build_kn_lut(E_grid, mu_grid, savelut = True):
    """Builds the lookup tables for various energies on a mu mesh

    Args:
        E_grid (nparray): grid of energy values to evaluate the k-n formula on [keV]
        mu_grid (nparray): grid of cosine of angles to build the k-n pdf's
        savelut (bool, optional): save the lookuptable or not, defaults to True

    Returns:
        nparray, nparray: values of pdf's and cdf's for each energy on the angular grid
    """
    mu_grid = np.asarray(mu_grid)
    # sort + unique for safety
    mu_grid = np.unique(mu_grid)

    nE = len(E_grid)
    nmu = len(mu_grid)
    pdf = np.empty((nE, nmu), dtype=float)

    for i, E in enumerate(E_grid):
        pdf_row = kn(E, mu_grid)
        norm = np.trapezoid(pdf_row, mu_grid)
        pdf[i, :] = 0.0 if norm <= 0 else pdf_row / norm

    # CDF
    cdf = integrate.cumulative_trapezoid(pdf, mu_grid, axis=1, initial=0.0)

    # Normalize numerically
    endvals = cdf[:, -1]
    good = (endvals > 0) & np.isfinite(endvals)
    cdf[good] /= endvals[good][:, None]
    cdf[~good] = 0.0

    # exact boundaries
    cdf[:, 0] = 0.0
    cdf[:, -1] = 1.0

    if savelut:
        np.savez("kn_lut.npz", E_grid=E_grid, mu_grid=mu_grid, pdf=pdf, cdf=cdf)

    return pdf, cdf

def sample_kn(E_ph, E_grid, mu_grid, cdf):
    """Samples the Klein-Nishima formula for photons of different energies
    Args:
        E_ph (nparray): (N,) photon energies [keV]
        E_grid (nparray): (nE,) energy grid [keV]
        theta_grid (nparray): (ntheta,) theta grid [rad]
        cdf (nparray): (nE, ntheta) lookuptable cdf's

    Returns:
        nparray, nparray: (N,), (N,) scattering angles and fractional weights sampled for each photon [rad]
    """
    n = len(E_ph)

    u = np.random.uniform(0,1,n)
    idx = np.searchsorted(E_grid, E_ph)  # returns index where E_ph would be inserted
    idx = np.clip(idx, 1, len(E_grid)-1)

    # map to nearest by comparing left/right distance:
    left = idx - 1
    right = idx
    # decide choose left or right based on which energy is closer
    choose_right = np.abs(E_grid[right] - E_ph) < np.abs(E_grid[left] - E_ph)
    bins = np.where(choose_right, right, left)  # final bin index per photon

    mu_out = np.zeros(n)
    F_allowed_arr = np.ones(n)

    unique_bins = np.unique(bins)
    for b in unique_bins:
        sel = (bins == b)
        cdf_b = cdf[b, :]
        mu_b = mu_grid

        # invert CDF
        u_local = u[sel]  # full range
        mu_vals = np.interp(u_local, cdf_b, mu_b)
        mu_out[sel] = mu_vals
    return mu_out, F_allowed_arr

PDF, CDF = build_kn_lut(E_GRID, MU_GRID, savelut=True)