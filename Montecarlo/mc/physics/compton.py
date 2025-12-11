import numpy as np

from mc.config import ME

def compton(E, mu):
    """Compton scattering formula

    Args:
        E (nparray): initial photon energies [keV]
        mu (nparray): cosine of scattering angles

    Returns:
        nparray: photon energies after Compton scattering [keV]
    """
    denom = 1 + (E*(1-mu)/ME)
    return E/denom