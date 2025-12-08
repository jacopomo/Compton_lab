import numpy as np

from mc.core.photon import Photons
from mc.utils.math3d import generate_random_directions

def test_scatter_update_dirs():
    n = 100
    positions = np.random.uniform(0,1,size=(n,3))
    directions = generate_random_directions(n)
    energies = np.random.uniform(1,1500, n)

    ps = Photons(0)
    ps.append(positions, directions, energies)

    assert np.max(ps.direc) <= 1
    assert np.max(ps.direc) >= 0

    assert np.allclose(np.linalg.norm(ps.direc, axis=1), np.ones(n), rtol=1e-9, atol=1e-9)

    ps.scatter_update_dirs(np.random.uniform(0,np.pi, n)) # scatter and see if direction vectors are still unitary

    assert np.max(ps.direc) <= 1
    assert np.max(ps.direc) >= 0

    assert np.allclose(np.linalg.norm(ps.direc, axis=1), np.ones(n), rtol=1e-9, atol=1e-9)