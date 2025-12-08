import numpy as np

def unpack_stacked(m):
    """
    Takes an Nx3 numpy array and unpacks it by column
    
    :param m: Nx3 numpy array to unpack
    :type m: numpy.ndarray
    
    :return: Unpacked columns of m
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
    """
    assert m.shape[1] == 3, "Please insert a Nx3 matrix to unpack"
    return m[:, 0], m[:, 1], m[:, 2]

def to_full_array(param, n):
    """Helper function that takes a parameter, scalar or nparray, 
    and returns a 'full' nparray of the desired length. If the parameter is
    scalar, np.full(n, param) is executed. If it's a nparray, the correct length is
    asserted and the correct array is passed

    Args:
        param (scalar, nparray (n,)): parameter to chech and convert
        n (int): desired length

    Returns:
        nparray: nparray (n,) 
    """
    is_scalar = np.isscalar(param)
    
    if not is_scalar:
        assert len(param) == n, f"Input array must have length {n}."
        
    return np.full(n, param, dtype=float) if is_scalar else param

def rotate_by_phi(points, angle):
    """_summary_

    Args:
        points (nparray): (N,3) N (x,y,z) points to rotate
        angle (float): angle to rotate the points by about the x-axis clockwise [radians]

    Returns:
        nparray: (N,3) rotated points
    """
    c = np.cos(angle)
    s = np.sin(angle)

    rotation_matrix = np.array([
        [1, 0,  0],
        [0, c,  s],
        [0, -s, c]
    ])
    rotated_points = points @ rotation_matrix.T
    return rotated_points

def generate_random_directions(n, xlim=(-1,1), ylim=(-1,1), zlim=(-1,1)):
    """Generates a (n,3) nparray of random direction unit vectors, with option to specify generation range
    Args:
        n (int): number of directions to generate
        xlim (tuple, optional): range in the x-direction. Defaults to (-1,1).
        ylim (tuple, optional): range in the y-direction. Defaults to (-1,1).
        zlim (tuple, optional): range in the z-direction. Defaults to (-1,1).

    Returns:
        nparray: (n,3) direction unit vectors
    """
    dx = np.random.uniform(xlim[0], xlim[1], n)
    dy = np.random.uniform(ylim[0], ylim[1], n)
    dz = np.random.uniform(zlim[0], zlim[1], n)
    directions = np.stack((dx,dy,dz),axis=-1)
    directions = directions/np.linalg.norm(directions, axis=1, keepdims=True)
    return directions