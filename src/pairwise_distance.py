# src/pairwise_distance.py

import cupy as cp


def pairwise_distance3(a, dims):
    """Computes the pairwise distances between two sets of points in a periodic box.

    Note: Ensure that the input array `a` is a (M,N,2)-shaped array before 
    calling this function, as it will not check the shape of the input array
    to avoid unnecessary overhead.

    Parameters
    ----------
    a : array-like (cupy.ndarray)
        (M,N,2)-shaped array that contains the xy-coordinates of the 
        N nearest neighbors of each of the M points.
    dims : array-like (cupy.ndarray)
        The dimensions of the periodic box.

    Returns
    -------
    dist : array-like (cupy.ndarray)
        (M,N)-shaped array that contains the pairwise distances between the 
        M points and their N nearest neighbors.
    """
    dist = cp.empty((a.shape[0], a.shape[1], a.shape[0], 2), dtype='float64')
    dist = cp.abs(cp.subtract(a[:,:,None], a[:,0,:], out=dist), out=dist)
    dist = cp.minimum(dist, dims-dist, out=dist)
    return cp.linalg.norm(dist, axis=3)