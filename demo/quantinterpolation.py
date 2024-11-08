# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Module that provides the same 'convolution' algorithm B as in the regular interpolation
module, up to the fact that the floating point number quantization degree is exposed
in the API and thus can be freely chosen.

Created on Sun Jun 5 2022, 11:31:27
@author: Bruno Zürcher
"""

from quantization import quant

from math import exp
import numpy as np

from numba import njit

from fastbarnes import interpolation


###############################################################################

def barnes(pts, val, sigma, x0, step, size, quant_bits, num_iter=4, max_dist=3.5):
    """
    Computes the 'convolution' Barnes interpolation for observation values `val` taken
    at sample points `pts` using Gaussian weights for the width parameter `sigma`.
    The underlying grid embedded in a Euclidean space is given with start point
    `x0`, regular grid steps `step` and extension `size`.

    Parameters
    ----------
    pts : numpy ndarray
        A 2-dimensional array of size N x 2 containing the x- and y-coordinates
        (or if you like the longitude/latitude) of the N sample points.
    val : numpy ndarray
        A 1-dimensional array of size N containing the N observation values.
    sigma : float
        The Gaussian width parameter to be used.
    x0 : numpy ndarray
        A 1-dimensional array of size 2 containing the coordinates of the
        start point of the grid to be used.
    step : float
        The regular grid point distance.
    size : tuple of 2 int values
        The extension of the grid in x- and y-direction.
    quant_bits : int
        The number of quantization bits, i.e. the number of least significant bits
        in the resulting field that are dropped.
    num_iter : int, optional
        The number of performed self-convolutions of the underlying rect-kernel.
        The default is 4.
    max_dist : float, optional
        The maximum distance between a grid point and the next sample point for which
        the Barnes interpolation is still calculated. Specified in sigma distances.
        The default is 3.5, i.e. the maximum distance is 3.5 * sigma.

    Returns
    -------
    numpy ndarray
        A 2-dimensional array containing the resulting field of the performed
        Barnes interpolation.
    """

    # perform simplified argument checking
    dim = pts.shape[1]

    # since we will modify the input array val in method _normalize_values(), we store a copy of it
    val = val.copy()

    # check sigma
    if isinstance(sigma, (list, tuple, np.ndarray)):
        if len(sigma) != dim:
            raise RuntimeError('specified sigma with invalid length: ' + str(len(sigma)))
        sigma = np.asarray(sigma, dtype=np.float64)
    else:
        sigma = np.full(dim, sigma, dtype=np.float64)
    # sigma is now a numpy array of length dim

    # check x0
    if isinstance(x0, (list, tuple, np.ndarray)):
        if len(x0) != dim:
            raise RuntimeError('specified x0 with invalid length: ' + str(len(x0)))
        x0 = np.asarray(x0, dtype=np.float64)
    else:
        x0 = np.full(dim, x0, dtype=np.float64)
    # x0 is now a numpy array of length dim

    # check step
    if isinstance(step, (list, tuple, np.ndarray)):
        if len(step) != dim:
            raise RuntimeError('specified step with invalid length: ' + str(len(step)))
        step = np.asarray(step, dtype=np.float64)
    else:
        step = np.full(dim, step, dtype=np.float64)
    # step is now a numpy array of length dim

    # check size
    if isinstance(size, (list, tuple, np.ndarray)):
        if len(size) != dim:
            raise RuntimeError('specified size with invalid length: ' + str(len(size)))
        size = tuple(size)
    elif dim != 1:
        raise RuntimeError('array size should be array-like of length: ' + str(dim))
    else:
        size = (size, )
    # size is now a tuple of length dim

    # compute weight that corresponds to specified max_dist
    max_dist_weight = exp(-max_dist**2/2)

    return _interpolate_quant_convol(pts, val, sigma, x0, step, size, quant_bits, num_iter, max_dist_weight)
    

#------------------------------------------------------------------------------

@njit
def _interpolate_quant_convol(pts, val, sigma, x0, step, size, quant_bits, num_iter, max_dist_weight):
    """ 
    Implements algorithm B presented in section 4 of the paper.
    In contrast to the _interpolate_convol() function found in ordinary interpolation
    module that applies a fixed quantization of 29 bits, this function allows the
    caller to freely choose the number of quantization bits.
    """
    offset = interpolation._normalize_values(val)

    # the grid fields to store the convolved values and weights
    rsize = size[::-1]
    vg = np.zeros(rsize, dtype=np.float64)
    wg = np.zeros(rsize, dtype=np.float64)

    # inject obs values into grid
    interpolation._inject_data_2d(vg, wg, pts, val, x0, step, size)

    # prepare convolution
    half_kernel_size = interpolation._get_half_kernel_size(sigma, step, num_iter)
    kernel_size = 2*half_kernel_size + 1

    # execute algorithm B
    interpolation._convolve_2d(vg, wg, sigma, step, size, kernel_size, num_iter, max_dist_weight)

    # yet to be considered:
    # - add offset again to resulting quotient
    vg = vg / wg + offset

    ### HERE DIFFERS IMPLEMENTATION ###
    # - and apply quantization operation: remove specified number of quantization bits
    quant(vg, quant_bits)

    return vg
