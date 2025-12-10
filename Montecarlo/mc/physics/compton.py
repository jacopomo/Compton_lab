import numpy as np

from mc.config import ME

def compton(E, theta):
    """Compton scattering formula

    Args:
        E (nparray): initial photon energies [keV]
        theta (nparray): scattering angles [rads]

    Returns:
        nparray: photon energies after Compton scattering [keV]
    """
    denom = 1 + (E*(1-np.cos(theta))/ME)
    return E/denom

def theta_min_threshold(E, E_thresh):
    """Formula to find the minimum scattering angle to deposit more
    energy than a selected threshold

    Args:
        E (nparray): photon energies
        E_thresh (nparray): energy thresholds

    Returns:
        nparray: minimum scattering angles [radians]
    """
    theta_min = np.empty_like(E, dtype=float)

    valid = E_thresh < E
    ratio = np.zeros_like(E)
    ratio[valid] = ME / E[valid] * (E_thresh[valid] / (E[valid] - E_thresh[valid]))
    # avoid invalid arccos
    ratio[valid] = np.clip(1 - ratio[valid], -1.0, 1.0)
    
    theta_min[valid] = np.arccos(ratio[valid])
    theta_min[~valid] = np.pi  # unphysical, cannot deposit that much energy
    return theta_min