import numpy as np

from mc.utils.math3d import unpack_stacked, to_full_array, rotate_by_phi

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
    
    assert np.all(full1 == np.full(n, 1)) # test for scalars that I pass

    full1 = to_full_array(np.full(n,1),n)

    assert np.all(full1 == np.full(n, 1)) # test for arrays that I pass

def test_rotate_by_phi():
    n = 100
    a, b, c = np.random.uniform(0,1,n), np.random.uniform(0,1,n), np.random.uniform(0,1,n)
    points = np.stack((a,b,c), axis=-1)

    rotated_points = rotate_by_phi(points, np.radians(20))
    
    x, y, z = unpack_stacked(rotated_points)

    assert np.all(x == a) # rotation does not change x-values
    assert np.all(y >= 0) # given initial points y should only be greater than 0
    assert np.all(z <= c) # given initial points z should only decrease
    assert np.all(y <= np.sqrt(2)) # given initial points y should only be less than root 2
    assert np.all(z <= np.sqrt(2)) # given initial points z should only be less than root 2

    un_rotate = rotate_by_phi(rotated_points, -np.radians(20))
    i, j, k = unpack_stacked(un_rotate)

    assert np.all(np.isclose(a, i, 1e-9)) # rotation and inverse = identity
    assert np.all(np.isclose(b, j, 1e-9)) # rotation and inverse = identity
    assert np.all(np.isclose(c, k, 1e-9)) # rotation and inverse = identity
