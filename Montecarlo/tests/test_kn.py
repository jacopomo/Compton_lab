import numpy as np

from mc.physics.kn_sampler import kn, build_kn_lut, sample_kn
from mc.config import RE

def test_kn():
    n = 10000
    energies = np.random.uniform(0,2000,n)
    mus = np.random.uniform(-1,1, n)
    diff_cross_sections = kn(energies, mus)
    assert np.all(diff_cross_sections) >= 0 # differential cross sections must be positive

    straights = np.ones(n)
    dcs_straights = kn(energies, straights)

    assert np.all(np.isclose(dcs_straights, RE**2, atol=1e-6)) # differential cross section for undeflected photons is RE^2

def test_build_kn_lut():
    E_grid = np.linspace(1,2000,50)
    mu_grid = np.linspace(-1,1,1000)
    pdf, cdf = build_kn_lut(E_grid, mu_grid, savelut=False)

    assert np.all(np.isclose(np.min(cdf, axis=1), 0, atol=1e-9)) # the cdf's start at 0
    assert np.all(np.isclose(np.max(cdf, axis=1), 1, atol=1e-9)) # the cdf's end at 1

def test_sample_kn():
    n = 500
    E_grid = np.linspace(1,2000,50)
    mu_grid = np.linspace(-1,1,1000)
    pdf, cdf = build_kn_lut(E_grid, mu_grid, savelut=False)
    
    E_ph = np.random.uniform(1,1000, n)
    mus, Fs = sample_kn(E_ph, E_grid, mu_grid, cdf)

    assert min(mus) >= -1.0
    assert max(mus) <= 1.0
    assert np.all(np.isclose(Fs, 1, atol=1e-9))