import numpy as np

from mc.physics.kn_sampler import kn, build_kn_lut, sample_kn
from mc.config import RE

def test_kn():
    n = 10000
    energies = np.random.uniform(0,2000,n)
    angles = np.random.uniform(-np.pi,np.pi, n)
    diff_cross_sections = kn(energies, angles)
    assert np.all(diff_cross_sections) >= 0 # differential cross sections must be positive

    straights = np.zeros(n)
    dcs_straights = kn(energies, straights)

    mask = dcs_straights == RE**2
    assert np.all(mask) # differential cross section for undeflected photons is RE^2

def test_build_kn_lut():
    E_grid = np.linspace(1,2000,50)
    theta_grid = np.linspace(0,2*np.pi,100)
    pdf, cdf = build_kn_lut(E_grid, theta_grid, savelut=False)

    assert np.all(np.isclose(np.min(cdf, axis=1), 0, atol=1e-9)) # the pdf's start at 0
    assert np.all(np.isclose(np.max(cdf, axis=1), 1, atol=1e-9)) # the cdf's end at 1

def test_sample_kn():
    n = 500
    theta_low = np.radians(20)
    theta_high = np.radians(170)
    E_grid = np.linspace(1,2000,50)
    theta_grid = np.linspace(0,2*np.pi,100)
    pdf, cdf = build_kn_lut(E_grid, theta_grid, savelut=False)
    
    E_ph = np.random.uniform(1,1000, n)
    angles, Fs = sample_kn(E_ph, E_grid, theta_grid, cdf, theta_low, theta_high)

    assert min(angles) >= theta_low  # test angular cut-off
    assert max(angles) <= theta_high # test angular cut-off

    assert min (Fs) > 0 # fractions must be greater than 0
    assert max(Fs) <= 1 # fractions must be less than or greater to 1
