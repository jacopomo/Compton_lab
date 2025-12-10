# mc/physics/kn_sampler.py

import numpy as np
import scipy.integrate as integrate

from mc.physics.compton import compton
from mc.config import RE, E1, E2, E_GRID, THETA_GRID
from mc.utils.math3d import to_full_array

def kn(E, theta):
    """Klein-Nishima differential cross section formula

    Args:
        E (nparray): (n,) initial photon energies [keV]
        theta (nparray): (n,) scattering angles [rads]

    Returns:
        nparray: (n,) differential cross section [cm^2]
    """
    lr = compton(E, theta) / E
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 0.5 * (RE**2) * (lr**2) * (lr + 1.0/lr - np.sin(theta)**2)
    out = np.where(np.isfinite(out), out, 0.0)
    return np.clip(out, 0.0, None)  

def build_kn_lut(E_grid, theta_grid, savelut = True):
    """Builds the lookup tables for various energies on a theta mesh

    Args:
        E_grid (nparray): grid of energy values to evaluate the k-n formula on [keV]
        theta_grid (nparray): grid of angles to build the k-n pdf's [rad]
        savelut (bool, optional): save the lookuptable or not, defaults to True

    Returns:
        nparray, nparray: values of pdf's and cdf's for each energy on the angular grid
    """
    theta_grid = np.asarray(theta_grid)
    mu_grid = np.cos(theta_grid)
    order = np.argsort(mu_grid)
    mu_grid = mu_grid[order]
    theta_grid = theta_grid[order]

    mu_grid, uniq = np.unique(mu_grid, return_index=True)
    theta_grid = theta_grid[uniq]

    nE = len(E_grid)
    nmu = len(mu_grid)
    pdf = np.empty((nE, nmu), dtype=float)

    theta_grid = np.arccos(mu_grid)

    for i, E in enumerate(E_grid):
        pdf_row = kn(E, theta_grid)
        
        norm = np.trapezoid(pdf_row, mu_grid)
        if norm <= 0:
            pdf[i, :] = 0.0
        else:
            pdf[i, :] = pdf_row / norm
    
    cdf = integrate.cumulative_trapezoid(pdf, mu_grid, axis=1, initial=0.0)
    cdf_end = cdf[:, -1]

    good = (cdf_end > 0) & np.isfinite(cdf_end)
    cdf[good] /= cdf_end[good][:, None]
    cdf[~good] = 0.0

    # exact boundaries
    cdf[:, 0] = 0.0
    cdf[:, -1] = 1.0

    if savelut:
        np.savez("kn_lut.npz", E_grid=E_grid, mu_grid=mu_grid, theta_grid=theta_grid, pdf=pdf, cdf=cdf)
    return pdf, cdf

def sample_kn(E_ph, E_grid, mu_grid, cdf, theta_low=0.0, theta_high=np.pi):
    """Samples the Klein-Nishima formula for photons of different energies
    Args:
        E_ph (nparray): (N,) photon energies [keV]
        E_grid (nparray): (nE,) energy grid [keV]
        theta_grid (nparray): (ntheta,) theta grid [rad]
        cdf (nparray): (nE, ntheta) lookuptable cdf's
        theta_low (float, optional): minimum scatter angle to sample from. Defaults to 0.0
        theta_high (float, optional): maximum scatter angle to sample from. Defaults to 2*np.pi

    Returns:
        nparray, nparray: (N,), (N,) scattering angles and fractional weights sampled for each photon [rad]
    """
    n = len(E_ph)
    theta_low = to_full_array(theta_low,n)
    theta_high = to_full_array(theta_high,n)

    mu_low = np.cos(theta_high)
    mu_high = np.cos(theta_low)

    u = np.random.uniform(0,1,n)
    idx = np.searchsorted(E_grid, E_ph)  # returns index where E_ph would be inserted
    idx = np.clip(idx, 1, len(E_grid)-1)

    # map to nearest by comparing left/right distance:
    left = idx - 1
    right = idx
    # decide choose left or right based on which energy is closer
    choose_right = np.abs(E_grid[right] - E_ph) < np.abs(E_grid[left] - E_ph)
    bins = np.where(choose_right, right, left)  # final bin index per photon

    theta_out = np.zeros(n)
    F_allowed_arr = np.zeros(n)

    unique_bins = np.unique(bins)
    for b in unique_bins:
        sel = (bins == b)
        cdf_b = cdf[b, :]
        mu_b = mu_grid

        # compute cdf at mu limits
        cdf_low = np.interp(mu_low[sel], mu_b, cdf_b)   
        cdf_high = np.interp(mu_high[sel], mu_b, cdf_b)
        F_allowed = cdf_high - cdf_low
        
        zero_mask = F_allowed <= 0
        if np.any(zero_mask):
            theta_out[sel][zero_mask] = np.pi
            F_allowed_arr[sel][zero_mask] = 0.0
            # continue on the rest

        ok_idx = ~zero_mask
        if np.any(ok_idx):
            u_sel = u[sel]
            # map u to local cdf segment
            u_local = cdf_low + u_sel * F_allowed
            # invert: find mu
            mu_vals = np.interp(u_local, cdf_b, mu_b)
            # convert to theta
            theta_vals = np.arccos(mu_vals)
            theta_out[sel] = theta_vals
            F_allowed_arr[sel] = F_allowed
    return theta_out, F_allowed_arr

PDF, CDF = build_kn_lut(E_GRID, THETA_GRID, savelut=True)