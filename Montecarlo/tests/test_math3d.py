import numpy as np

from mc.utils.math3d import unpack_stacked, to_full_array

def test_unpack_stacked():
    n = 100
    a, b, c = np.random.uniform(0,1,n), np.random.uniform(0,1,n), np.random.uniform(0,1,n)
    arr = np.stack((a,b,c), axis=-1)

    x, y, z = unpack_stacked(arr)

    assert len(x) == n
    assert len(y) == n
    assert len(z) == n

def test_to_full_array():
    n = 100
    full1 = to_full_array(1,n)
    
    assert np.all(full1 == np.full(n, 1)) # Test for scalars that I pass

    full1 = to_full_array(np.full(n,1),n)

    assert np.all(full1 == np.full(n, 1)) # Test for arrays that I pass