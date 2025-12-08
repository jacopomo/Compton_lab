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