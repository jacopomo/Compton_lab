import numpy as np

from mc.physics.kn_sampler import kn
from mc.config import RE

def test_kn():
    n = 10000
    energies = np.random.uniform(0,2000,n)
    angles = np.random.uniform(-np.pi,np.pi, n)
    diff_cross_sections = kn(energies, angles)
    assert np.all(diff_cross_sections) >= 0 # Differential cross sections must be positive

    straights = np.zeros(n)
    dcs_straights = kn(energies, straights)

    mask = dcs_straights == RE**2
    assert np.all(mask) # Differential cross section for undeflected photons is RE^2