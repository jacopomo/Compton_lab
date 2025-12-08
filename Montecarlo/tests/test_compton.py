import numpy as np

from mc.physics.compton import compton

def test_compton_formula():
    n = 10000
    energies = np.random.uniform(0,2000,n)
    angles = np.random.uniform(-np.pi,np.pi, n)
    e_finals = compton(energies, angles)
    assert np.all(e_finals) >= 0 # Compton energy must still be positive

    assert len(e_finals) == n 
    mask = e_finals <= energies

    assert np.all(mask) # Compton energy must be less than original energy 

    straights = np.zeros(n)
    e_straights = compton(energies, straights)

    mask = e_straights == energies
    assert np.all(mask) # Compton scattering at 0 degrees must not change the energy