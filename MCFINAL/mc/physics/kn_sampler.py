import numpy as np
import scipy.integrate as integrate

from mc.physics.compton import compton
from mc.config import RE
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
    return 0.5*(RE**2)*(lr)**2*(lr + (lr)**-1 - (np.sin(theta))**2)  

def build_kn_lut(E_grid, theta_grid):
    """Builds the lookup tables for various energies on a theta mesh

    Args:
        E_grid (nparray): grid of energy values to evaluate the k-n formula on [keV]
        theta_grid (nparray): grid of angles to build the k-n pdf's [rad]

    Returns:
        nparray, nparray: values of pdf's and cdf's for each energy on the angular grid
    """
    nE = len(E_grid)
    nmu = len(theta_grid)
    pdf = np.empty((nE, nmu), dtype=float)
    for i, E in enumerate(E_grid):
        pdf_row = kn(E, theta_grid)
        
        norm = np.trapz(pdf_row, theta_grid)
        pdf[i, :] = pdf_row / norm

    cdf = integrate.cumulative_trapezoid(pdf, theta_grid, axis=1, initial=0.0)
    np.savez("kn_lut.npz", E_grid=E_grid, theta_grid=theta_grid, pdf=pdf, cdf=cdf)
    return pdf, cdf

def sample_kn(E_ph, E_grid, theta_grid, cdf, theta_low=0.0, theta_high=2*np.pi):
    """Samples the Klein-Nishima formula for photons of different energies
    Args:
        E_ph (nparray): (N,) photon energies [keV]
        E_grid (nparray): (nE,) energy grid [keV]
        theta_grid (nparray): (ntheta,) theta grid [rad]
        cdf (nparray): (nE, ntheta) lookuptable cdf's
        theta_low (float, optional): minimum scatter angle to sample from. Defaults to 0.0
        theta_high (float, optional): maximum scatter angle to sample from. Defaults to 2*np.pi

    Returns:
        nparray: (N,) scattering angles sampled for each photon [rad]
    """
    n = len(E_ph)
    theta_low = to_full_array(theta_low,n)
    theta_high = to_full_array(theta_high,n)

    u = np.random.uniform(0,1,n)
    idx = np.searchsorted(E_grid, E_ph)  # returns index where E_ph would be inserted
    # correct indices at edges
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
        # compute cdf at mu limits
        cdf_low = np.interp(theta_low[sel], theta_grid, cdf_b)   
        cdf_high = np.interp(theta_high[sel], theta_grid, cdf_b)
        F_allowed = cdf_high - cdf_low
        # map u_base in [0,1] to [cdf_low, cdf_high]
        u_sel = u[sel]
        u_local = cdf_low + u_sel * F_allowed
        # invert: get mu
        theta_vals = np.interp(u_local, cdf_b, theta_grid)
        theta_out[sel] = theta_vals
        F_allowed_arr[sel] = F_allowed
    return theta_out, F_allowed_arr