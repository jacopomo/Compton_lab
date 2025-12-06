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
