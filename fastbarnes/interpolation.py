# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Module that provides different Barnes interpolation algorithms that use
the distance metric of the Euclidean plane.
To attain competitive performance, the code is written using Numba's
just-in-time compiler and thus has to use the respective programming idiom,
which is sometimes not straightforward to read at a first glance. Allocated
memory is as far as possible reused in order to reduce the workload imposed
on the garbage collector.

Created on Sat May 14 2022, 13:10:47
@author: Bruno Zürcher
"""

from math import sqrt, exp, log, pi
import numpy as np

from numba import njit

from fastbarnes.util import kdtree


###############################################################################

def barnes(pts, val, sigma, x0, step, size, method='optimized_convolution',
           num_iter=4, max_dist=3.5, min_weight=0.001):
    """
    Computes the Barnes interpolation for observation values `val` taken at sample
    points `pts` using Gaussian weights for the width parameter `sigma`.
    The underlying grid embedded in an M-dimensional Euclidean space is given with
    start point `x0`, regular grid steps `step` and extension `size`.
    The implementation supports Euclidean spaces of dimension M == 1, 2 or 3.

    Parameters
    ----------
    pts : numpy ndarray
        A 2-dimensional array of size N x M containing the (x_1, ..., x_M)
        coordinates of the N sample points.
        For the two-dimensional case M == 2, this could for instance be the
        N x 2 array of longitude/latitude coordinate pairs of the sample points.
        If only a 1-dimensional array is specified, it is assumed that we act
        on a Euclidean space of dimension M == 1 and the length of the array
        corresponds to the number of sample points N.
    val : numpy ndarray
        A 1-dimensional array of size N containing the N observation values.
    sigma : float or numpy ndarray
        The Gaussian width parameter to be used. Either a scalar - in which case
        the same sigma is applied in each direction - or a 1-dimensional array of
        size M specifying the sigmas in each direction separately.
    x0 : numpy ndarray
        A 1-dimensional array of size M containing the coordinates of the
        start point of the grid to be used.
    step : float or numpy ndarray
        The regular grid point distance. Either a scalar - in which case the
        same grid step is used for each direction - or a 1-dimensional array of
        size M specifying the steps for each direction separately.
    size : tuple or numpy ndarray of M int values
        The extension of the grid in each direction.
    method : {'optimized_convolution', 'convolution', 'radius', 'naive'}
        Designates the Barnes interpolation method to be used. The possible
        implementations that can be chosen are 'naive' for the straightforward
        implementation (algorithm A from paper), 'radius' to consider only sample
        points within a specific radius of influence, both with an algorithmic
        complexity of O(N x W x H). Method 'radius' can only be applied in
        dimension 2.
        The choice 'convolution' implements algorithm B specified in the paper
        and 'optimized_convolution' is its optimization by appending tail values
        to the rectangular kernel. The latter two algorithms reduce the complexity
        down to O(N + W x H).
        The default is 'optimized_convolution'.
    num_iter : int, optional
        The number of performed self-convolutions of the underlying rect-kernel.
        Applies only if method is 'optimized_convolution' or 'convolution'.
        The default is 4.
    max_dist : float, optional
        The maximum distance between a grid point and the next sample point for which
        the Barnes interpolation is still calculated. Specified in sigma distances.
        Applies only if method is 'optimized_convolution' or 'convolution'.
        The default is 3.5, i.e. the maximum distance is 3.5 * sigma.
    min_weight : float, optional
        Choose radius of influence such that Gaussian weight of considered sample
        points is greater than `min_weight`.
        Applies only if method is 'radius'. Recommended values are 0.001 and less.
        The default is 0.001, which corresponds to a radius of 3.717 * sigma.

    Returns
    -------
    numpy ndarray
        An M-dimensional array containing the resulting values of the performed
        Barnes interpolation.
        Note: The index mapping of the resulting array is [y, x] in case of 2D grids
        and [z, y, x] for 3D grids.
        This order complies to the quasi-standard and allows efficient access to
        (y, x)-slices in the case of volumetric data.
    """

    # check structure of sample point coordinates array pts
    if not isinstance(pts, np.ndarray):
        raise RuntimeError('specified pts is not a numpy ndarray')
    if len(pts.shape) > 2:
        raise RuntimeError('expected pts array of shape (N, M) but was: ' + str(pts.shape))
    if len(pts.shape) == 1:
        # assuming we deal implicitly with case M == 1, i.e. reshape to an N x 1 array
        pts = np.reshape(pts, (-1, 1))
    # now pts array has shape N x M

    dim = pts.shape[1]
    if dim < 1 or dim > 3:
        raise RuntimeError('Barnes interpolation supports only sample points in dimensions 1, 2 or 3')

    # check structure of sample values array val
    if not isinstance(val, np.ndarray):
        raise RuntimeError('specified val is not a numpy ndarray')
    if len(val.shape) > 1:
        raise RuntimeError('expected val array of shape (N) but was: ' + str(val.shape))
    if val.shape[0] != pts.shape[0]:
        raise RuntimeError('pts and val arrays have inconsistent shapes: ' + str(pts.shape) + ' vs. ' + str(val.shape))

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

    if method == 'optimized_convolution':
        # check size of resulting rectangular kernel against size of grid
        kernel_size = 2*_get_half_kernel_size_opt(sigma, step, num_iter) + 1
        for m in range(dim):
            if kernel_size[m] >= size[m]:
                raise RuntimeError('resulting rectangular kernel size should be smaller w.r.t. specified grid: '
                    + str(kernel_size) + ' vs. ' + str(size))
        return _interpolate_opt_convol(pts, val, sigma, x0, step, size, num_iter, max_dist_weight)
        
    elif method == 'convolution':
        # check size of resulting rectangular kernel against size of grid
        kernel_size = 2*_get_half_kernel_size(sigma, step, num_iter) + 1
        for m in range(dim):
            if kernel_size[m] >= size[m]:
                raise RuntimeError('resulting rectangular kernel size should be smaller w.r.t. specified grid: '
                    + str(kernel_size) + ' vs. ' + str(size))
        return _interpolate_convol(pts, val, sigma, x0, step, size, num_iter, max_dist_weight)
    
    elif method == 'radius':
        # specific checks for radius algorithm, which works only in 2D and scalar sigma
        if dim != 2:
            raise RuntimeError('radius algorithm works only in 2D but data is: ' + str(dim) + 'D')
        if sigma[0] != sigma[1]:
            raise RuntimeError('radius algorithm in 2D works only for scalar sigma value but sigma is: ' + str(sigma))
        return _interpolate_radius(pts, val, sigma[0], x0, step, size, max_dist_weight, min_weight)
        
    elif method == 'naive':
        return _interpolate_naive(pts, val, sigma, x0, step, size)
        
    else:
        raise RuntimeError("encountered invalid Barnes interpolation method: " + method)
    

# -----------------------------------------------------------------------------

@njit
def _normalize_values(val):
    """
    Offsets the observation values such that they are centered over 0.
    """
    offset = (np.amin(val) + np.amax(val)) / 2.0
    # center range of observation values around 0
    val -= offset
    return offset


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

@njit
def _inject_data_1d(vg, wg, pts, val, x0, step, size):
    """
    Injects the observations values and weights, respectively, into the
    corresponding fields as described by algorithm B.1.
    """
    for k in range(len(pts)):
        xc = (pts[k,0]-x0[0]) / step[0]
        if xc < 0.0 or xc >= size[0]-1:
            continue
        xi = int(xc)
        xw = xc - xi

        w = (1.0-xw)
        vg[xi] += w*val[k]
        wg[xi] += w

        w = xw
        vg[xi+1] += w*val[k]
        wg[xi+1] += w


@njit
def _inject_data_2d(vg, wg, pts, val, x0, step, size):
    """
    Injects the observations values and weights, respectively, into the
    corresponding fields as described by algorithm B.1.
    """
    for k in range(len(pts)):
        xc = (pts[k,0]-x0[0]) / step[0]
        yc = (pts[k,1]-x0[1]) / step[1]
        if xc < 0.0 or yc < 0.0 or xc >= size[0]-1 or yc >= size[1]-1:
            continue
        xi = int(xc)
        yi = int(yc)
        xw = xc - xi
        yw = yc - yi
        
        w = (1.0-xw)*(1.0-yw)
        vg[yi, xi] += w*val[k]
        wg[yi, xi] += w

        w = xw*(1.0-yw)
        vg[yi, xi+1] +=w*val[k]
        wg[yi, xi+1] += w
        
        w = xw*yw
        vg[yi+1, xi+1] += w*val[k]
        wg[yi+1, xi+1] += w
        
        w = (1.0-xw)*yw
        vg[yi+1, xi] += w*val[k]
        wg[yi+1, xi] += w


@njit
def _inject_data_3d(vg, wg, pts, val, x0, step, size):
    """
    Injects the observations values and weights, respectively, into the
    corresponding fields as described by algorithm B.1.
    """
    for k in range(len(pts)):
        xc = (pts[k,0]-x0[0]) / step[0]
        yc = (pts[k,1]-x0[1]) / step[1]
        zc = (pts[k,2]-x0[2]) / step[2]
        if xc < 0.0 or yc < 0.0 or zc < 0.0 or xc >= size[0]-1 or yc >= size[1]-1 or zc >= size[2]-1:
            continue
        xi = int(xc)
        yi = int(yc)
        zi = int(zc)
        xw = xc - xi
        yw = yc - yi
        zw = zc - zi

        w = (1.0-xw)*(1.0-yw)*(1.0-zw)
        vg[zi,yi,xi] += w*val[k]
        wg[zi,yi,xi] += w

        w = xw*(1.0-yw)*(1.0-zw)
        vg[zi,yi,xi+1] += w*val[k]
        wg[zi,yi,xi+1] += w

        w = xw*yw*(1.0-zw)
        vg[zi,yi+1,xi+1] += w*val[k]
        wg[zi,yi+1,xi+1] += w

        w = (1.0-xw)*yw*(1.0-zw)
        vg[zi,yi+1,xi] += w*val[k]
        wg[zi,yi+1,xi] += w

        w = (1.0-xw)*(1.0-yw)*zw
        vg[zi+1,yi,xi] += w*val[k]
        wg[zi+1,yi,xi] += w

        w = xw*(1.0-yw)*zw
        vg[zi+1,yi,xi+1] += w*val[k]
        wg[zi+1,yi,xi+1] += w

        w = xw*yw*zw
        vg[zi+1,yi+1,xi+1] += w*val[k]
        wg[zi+1,yi+1,xi+1] += w

        w = (1.0-xw)*yw*zw
        vg[zi+1,yi+1,xi] += w*val[k]
        wg[zi+1,yi+1,xi] += w


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

@njit
def _interpolate_opt_convol(pts, val, sigma, x0, step, size, num_iter, max_dist_weight):
    """ 
    Implements algorithm B presented in section 4 of the paper but optimized for
    a rectangular window with a tail value alpha.
    """
    offset = _normalize_values(val)

    # the grid fields to store the convolved values and weights - reverse grid dimensions
    rsize = size[::-1]
    vg = np.zeros(rsize, dtype=np.float64)
    wg = np.zeros(rsize, dtype=np.float64)
    
    # inject obs values into grid
    dim = len(size)
    if dim == 1:
        _inject_data_1d(vg, wg, pts, val, x0, step, size)
    elif dim == 2:
        _inject_data_2d(vg, wg, pts, val, x0, step, size)
    else:
        _inject_data_3d(vg, wg, pts, val, x0, step, size)
        
    # prepare convolution
    half_kernel_size = _get_half_kernel_size_opt(sigma, step, num_iter)
    kernel_size = 2*half_kernel_size + 1
    tail_value = _get_tail_value(sigma, step, num_iter)
    
    # execute algorithm B
    if dim == 1:
        _convolve_tail_1d(vg, wg, sigma, step, size, kernel_size, num_iter, tail_value, max_dist_weight)
    elif dim == 2:
        _convolve_tail_2d(vg, wg, sigma, step, size, kernel_size, num_iter, tail_value, max_dist_weight)
    else:
        _convolve_tail_3d(vg, wg, sigma, step, size, kernel_size, num_iter, tail_value, max_dist_weight)

    # yet to be considered for finalization:
    # - add offset again to resulting quotient
    # - and apply quantization operation:
    #   here by casting double to float and thus drop around 29 least significant bits
    return (vg / wg + offset).astype(np.float32)


# -----------------------------------------------------------------------------

@njit
def _convolve_tail_1d(vg, wg, sigma, step, size, kernel_size, num_iter, tail_value, max_dist_weight):
    """
    Computes the `num_iter`-fold convolution of the specified 1-dim arrays with the specified rect-kernel.
    """
    # convolve array in x-direction
    h_arr = np.empty(size[0], dtype=np.float64)
    # convolve array values
    vg[:] = _accumulate_tail_array(vg[:].copy(), h_arr, size[0], kernel_size[0], num_iter, tail_value[0])

    # convolve array weights
    wg[:] = _accumulate_tail_array(wg[:].copy(), h_arr, size[0], kernel_size[0], num_iter, tail_value[0])

    # compute limit wg array value for which weight > max_dist_weight, i.e. grid points with greater distance
    #   than max_dist*sigma will evaluate to NaN
    # since we dropped common factors in our computation, we have to revert their cancellation in the
    #   following computation
    conv_scale_factor = (kernel_size+2*tail_value)**num_iter / sqrt(2 * pi) / (sigma/step)
    conv_scale_factor = np.prod(conv_scale_factor) * max_dist_weight

    # set smaller weights to NaN with overall effect that corresponding quotient is NaN, too
    for i in range(size[0]):
        if wg[i] < conv_scale_factor: wg[i] = np.nan


@njit
def _convolve_tail_2d(vg, wg, sigma, step, size, kernel_size, num_iter, tail_value, max_dist_weight):
    """
    Computes the `num_iter`-fold convolution of each 1-dim sub-array and each direction with the specified rect-kernel.
    """
    # convolve rows in x-direction
    h_arr = np.empty(size[0], dtype=np.float64)
    for j in range(size[1]):
        # convolve row values
        vg[j,:] = _accumulate_tail_array(vg[j,:].copy(), h_arr, size[0], kernel_size[0], num_iter, tail_value[0])

        # convolve row weights
        wg[j,:] = _accumulate_tail_array(wg[j,:].copy(), h_arr, size[0], kernel_size[0], num_iter, tail_value[0])

    # convolve columns in y- direction
    h_arr = np.empty(size[1], dtype=np.float64)
    for i in range(size[0]):
        # convolve column values
        vg[:,i] = _accumulate_tail_array(vg[:,i].copy(), h_arr, size[1], kernel_size[1], num_iter, tail_value[1])

        # convolve column weights
        wg[:,i] = _accumulate_tail_array(wg[:,i].copy(), h_arr, size[1], kernel_size[1], num_iter, tail_value[1])

    # compute limit wg array value for which weight > max_dist_weight, i.e. grid points with greater distance
    #   than max_dist*sigma will evaluate to NaN
    # since we dropped common factors in our computation, we have to revert their cancellation in the
    #   following computation
    conv_scale_factor = (kernel_size+2*tail_value)**num_iter / sqrt(2 * pi) / (sigma/step)
    conv_scale_factor = np.prod(conv_scale_factor) * max_dist_weight

    # set smaller weights to NaN with overall effect that corresponding quotient is NaN, too
    for j in range(size[1]):
        for i in range(size[0]):
            if wg[j,i] < conv_scale_factor: wg[j,i] = np.nan


@njit
def _convolve_tail_3d(vg, wg, sigma, step, size, kernel_size, num_iter, tail_value, max_dist_weight):
    """
    Computes the `num_iter`-fold convolution of each 1-dim sub-array and each direction with the specified rect-kernel.
    """
    # convolve all 1d-sub-arrays in x-direction
    h_arr = np.empty(size[0], dtype=np.float64)
    for k in range(size[2]):
        for j in range(size[1]):
            # convolve array values
            vg[k,j,:] = _accumulate_tail_array(vg[k,j,:].copy(), h_arr, size[0], kernel_size[0], num_iter, tail_value[0])

            # convolve array weights
            wg[k,j,:] = _accumulate_tail_array(wg[k,j,:].copy(), h_arr, size[0], kernel_size[0], num_iter, tail_value[0])

    # convolve all 1d-sub-arrays in y-direction
    h_arr = np.empty(size[1], dtype=np.float64)
    for k in range(size[2]):
        for i in range(size[0]):
            # convolve array values
            vg[k,:,i] = _accumulate_tail_array(vg[k,:,i].copy(), h_arr, size[1], kernel_size[1], num_iter, tail_value[1])

            # convolve array weights
            wg[k,:,i] = _accumulate_tail_array(wg[k,:,i].copy(), h_arr, size[1], kernel_size[1], num_iter, tail_value[1])

    # convolve all 1d-sub-arrays in z-direction
    h_arr = np.empty(size[2], dtype=np.float64)
    for j in range(size[1]):
        for i in range(size[0]):
            # convolve array values
            vg[:,j,i] = _accumulate_tail_array(vg[:,j,i].copy(), h_arr, size[2], kernel_size[2], num_iter, tail_value[2])

            # convolve array weights
            wg[:,j,i] = _accumulate_tail_array(wg[:,j,i].copy(), h_arr, size[2], kernel_size[2], num_iter, tail_value[2])

    # compute limit wg array value for which weight > max_dist_weight, i.e. grid points with greater distance
    #   than max_dist*sigma will evaluate to NaN
    # since we dropped common factors in our computation, we have to revert their cancellation in the
    #   following computation
    conv_scale_factor = (kernel_size+2*tail_value)**num_iter / sqrt(2 * pi) / (sigma/step)
    conv_scale_factor = np.prod(conv_scale_factor) * max_dist_weight

    # set smaller weights to NaN with overall effect that corresponding quotient is NaN, too
    for k in range(size[2]):
        for j in range(size[1]):
            for i in range(size[0]):
                if wg[k,j,i] < conv_scale_factor: wg[k,j,i] = np.nan


# -----------------------------------------------------------------------------
    
@njit
def _accumulate_tail_array(in_arr, h_arr, arr_len, rect_len, num_iter, alpha):
    """
    Computes the `num_iter`-fold convolution of the specified 1-dim array ìn_arr`
    with a rect-kernel of length rect_len and tail values `alpha`. To obtain the
    actual convolution with a corresponding uniform distribution, the result would have
    to be scaled with a factor 1/rect_len^num_iter. But this scaling is not implemented,
    since these factors are canceled when the resulting fields are divided with
    each other.
    """
    # the half window size T
    h0 = (rect_len-1) // 2
    h0_1 = h0 + 1
    h1 = rect_len - h0
    for i in range(num_iter):
        # accumulates values under regular part of window (without tails!)
        accu = 0.0
        # phase a: window center still outside array
        # accumulate first h0 elements
        for k in range(-h0, 0):
            accu += in_arr[k+h0]
        # phase b: window center inside array but window does not cover array completely
        # accumulate remaining rect_len elements and write their value into array
        for k in range(0, h1):
            accu += in_arr[k+h0]
            h_arr[k] = accu + alpha*in_arr[k+h0_1]
        # phase c: window completely contained in array
        # add difference of border elements and write value into array
        for k in range(h1, arr_len-h0_1):
            accu += (in_arr[k+h0] - in_arr[k-h1])
            h_arr[k] = accu + alpha*(in_arr[k-h1]+in_arr[k+h0_1])
        # phase c': very last element
        k = arr_len-h0_1
        accu += (in_arr[k+h0] - in_arr[k-h1])
        h_arr[k] = accu + alpha*in_arr[k-h1]
        # phase d (mirroring phase b): window center still inside array but window does not cover array completely
        # de-accumulate elements and write value into array
        for k in range(arr_len-h0, arr_len):
            accu -= in_arr[k-h1]
            h_arr[k] = accu + alpha*in_arr[k-h1]
        # phase e (mirroring phase a): window center left array
        # unnecessary since value is not written

        # h_arr contains convolution result of this pass
        # swap arrays and start over next convolution
        h = in_arr
        in_arr = h_arr
        h_arr = h
    
    return in_arr


# -----------------------------------------------------------------------------

def _to_np(value):
    """ Converts a scalar to a numpy array. """
    return np.asarray([value], dtype=np.float64)


def get_half_kernel_size_opt(sigma, step, num_iter):
    """ Computes the half kernel size T for the optimized convolution algorithm. Version for scalar arguments. """
    return _get_half_kernel_size_opt(_to_np(sigma), _to_np(step), num_iter)[0]


@njit
def _get_half_kernel_size_opt(sigma, step, num_iter):
    """ Computes the half kernel size T for the optimized convolution algorithm. Version for numpy arrays arguments. """
    s = sigma / step
    return ((np.sqrt(1.0+12*s*s/num_iter) - 1.0) / 2.0).astype(np.int32)


def get_tail_value(sigma, step, num_iter):
    """ Computes the tail value alpha for the optimized convolution algorithm. Version for scalar arguments. """
    return _get_tail_value(_to_np(sigma), _to_np(step), num_iter)[0]


@njit
def _get_tail_value(sigma, step, num_iter):
    """ Computes the tail value alpha for the optimized convolution algorithm. """
    half_kernel_size = _get_half_kernel_size_opt(sigma, step, num_iter)
    kernel_size = 2*half_kernel_size + 1

    sigma_rect_sqr = (half_kernel_size+1)*half_kernel_size/3.0*step**2
    # slightly rearranged expression from equ. (12)
    return 0.5*kernel_size*(sigma**2/num_iter - sigma_rect_sqr) \
        / (((half_kernel_size+1)*step)**2 - sigma**2/num_iter)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

@njit
def _interpolate_convol(pts, val, sigma, x0, step, size, num_iter, max_dist_weight):
    """ 
    Implements algorithm B presented in section 4 of the paper.
    """
    offset = _normalize_values(val)

    # the grid fields to store the convolved values and weights - reverse grid dimensions
    rsize = size[::-1]
    vg = np.zeros(rsize, dtype=np.float64)
    wg = np.zeros(rsize, dtype=np.float64)
    
    # inject obs values into grid
    dim = len(size)
    if dim == 1:
        _inject_data_1d(vg, wg, pts, val, x0, step, size)
    elif dim == 2:
        _inject_data_2d(vg, wg, pts, val, x0, step, size)
    else:
        _inject_data_3d(vg, wg, pts, val, x0, step, size)

    # prepare convolution
    half_kernel_size = _get_half_kernel_size(sigma, step, num_iter)
    kernel_size = 2*half_kernel_size + 1
        
    # execute algorithm B
    if dim == 1:
        _convolve_1d(vg, wg, sigma, step, size, kernel_size, num_iter, max_dist_weight)
    elif dim == 2:
        _convolve_2d(vg, wg, sigma, step, size, kernel_size, num_iter, max_dist_weight)
    else:
        _convolve_3d(vg, wg, sigma, step, size, kernel_size, num_iter, max_dist_weight)

    # yet to be considered:
    # - add offset again to resulting quotient
    # - and apply quantization operation:
    #   here by temporary casting double to float and thus drop around 29 least significant bits
    return (vg / wg + offset).astype(np.float32)


# -----------------------------------------------------------------------------

@njit
def _convolve_1d(vg, wg, sigma, step, size, kernel_size, num_iter, max_dist_weight):
    """
    Computes the `num_iter`-fold convolution of the specified 1-dim sub-array with the specified rect-kernel.
    """
    # convolve array in x-direction
    h_arr = np.empty(size[0], dtype=np.float64)
    # convolve array values
    vg[:] = _accumulate_array(vg[:].copy(), h_arr, size[0], kernel_size[0], num_iter)

    # convolve array weights
    wg[:] = _accumulate_array(wg[:].copy(), h_arr, size[0], kernel_size[0], num_iter)

    # compute limit wg array value for which weight > max_dist_weight, i.e. grid points with greater distance
    #   than max_dist*sigma will evaluate to NaN
    # since we dropped common factors in our computation, we have to revert their cancellation in the
    #   following computation
    conv_scale_factor = kernel_size**num_iter / sqrt(2 * pi) / (sigma/step)
    conv_scale_factor = np.prod(conv_scale_factor) * max_dist_weight

    # set smaller weights to NaN with overall effect that corresponding quotient is NaN, too
    for i in range(size[0]):
        if wg[i] < conv_scale_factor: wg[i] = np.nan


@njit
def _convolve_2d(vg, wg, sigma, step, size, kernel_size, num_iter, max_dist_weight):
    """
    Computes the `num_iter`-fold convolution of each 1-dim sub-array and each direction with the specified rect-kernel.
    """
    # convolve rows in x-direction
    h_arr = np.empty(size[0], dtype=np.float64)
    for j in range(size[1]):
        # convolve row values
        vg[j,:] = _accumulate_array(vg[j,:].copy(), h_arr, size[0], kernel_size[0], num_iter)

        # convolve row weights
        wg[j,:] = _accumulate_array(wg[j,:].copy(), h_arr, size[0], kernel_size[0], num_iter)

    # convolve columns in y- direction
    h_arr = np.empty(size[1], dtype=np.float64)
    for i in range(size[0]):
        # convolve column values
        vg[:,i] = _accumulate_array(vg[:,i].copy(), h_arr, size[1], kernel_size[1], num_iter)

        # convolve column weights
        wg[:,i] = _accumulate_array(wg[:,i].copy(), h_arr, size[1], kernel_size[1], num_iter)

    # compute limit wg array value for which weight > max_dist_weight, i.e. grid points with greater distance
    #   than max_dist*sigma will evaluate to NaN
    # since we dropped common factors in our computation, we have to revert their cancellation in the
    #   following computation
    conv_scale_factor = kernel_size**num_iter / sqrt(2 * pi) / (sigma/step)
    conv_scale_factor = np.prod(conv_scale_factor) * max_dist_weight

    # set smaller weights to NaN with overall effect that corresponding quotient is NaN, too
    for j in range(size[1]):
        for i in range(size[0]):
            if wg[j,i] < conv_scale_factor: wg[j,i] = np.nan


@njit
def _convolve_3d(vg, wg, sigma, step, size, kernel_size, num_iter, max_dist_weight):
    """
    Computes the `num_iter`-fold convolution of each 1-dim sub-array and each direction with the specified rect-kernel.
    """
    # convolve all 1d-sub-arrays in x-direction
    h_arr = np.empty(size[0], dtype=np.float64)
    for k in range(size[2]):
        for j in range(size[1]):
            # convolve array values
            vg[k,j,:] = _accumulate_array(vg[k,j,:].copy(), h_arr, size[0], kernel_size[0], num_iter)

            # convolve array weights
            wg[k,j,:] = _accumulate_array(wg[k,j,:].copy(), h_arr, size[0], kernel_size[0], num_iter)

    # convolve all 1d-sub-arrays in y-direction
    h_arr = np.empty(size[1], dtype=np.float64)
    for k in range(size[2]):
        for i in range(size[0]):
            # convolve array values
            vg[k,:,i] = _accumulate_array(vg[k,:,i].copy(), h_arr, size[1], kernel_size[1], num_iter)

            # convolve array weights
            wg[k,:,i] = _accumulate_array(wg[k,:,i].copy(), h_arr, size[1], kernel_size[1], num_iter)

    # convolve all 1d-sub-arrays in z-direction
    h_arr = np.empty(size[2], dtype=np.float64)
    for j in range(size[1]):
        for i in range(size[0]):
            # convolve array values
            vg[:,j,i] = _accumulate_array(vg[:,j,i].copy(), h_arr, size[2], kernel_size[2], num_iter)

            # convolve array weights
            wg[:,j,i] = _accumulate_array(wg[:,j,i].copy(), h_arr, size[2], kernel_size[2], num_iter)

    # compute limit wg array value for which weight > max_dist_weight, i.e. grid points with greater distance
    #   than max_dist*sigma will evaluate to NaN
    # since we dropped common factors in our computation, we have to revert their cancellation in the
    #   following computation
    conv_scale_factor = kernel_size**num_iter / sqrt(2 * pi) / (sigma/step)
    conv_scale_factor = np.prod(conv_scale_factor) * max_dist_weight

    # set smaller weights to NaN with overall effect that corresponding quotient is NaN, too
    for k in range(size[2]):
        for j in range(size[1]):
            for i in range(size[0]):
                if wg[k,j,i] < conv_scale_factor: wg[k,j,i] = np.nan


# -----------------------------------------------------------------------------

@njit
def _accumulate_array(in_arr, h_arr, arr_len, rect_len, num_iter):
    """
    Computes the `num_iter`-fold convolution of the specified 1-dim array ìn_arr`
    with a rect-kernel of length rect_len. To obtain the actual convolution with
    a corresponding uniform distribution, the result would have to be scaled with
    a factor 1/rect_len^num_iter. But this scaling is not implemented, since these
    factors are canceled when the resulting fields are divided with each other.
    """
    # the half window size T
    h0 = (rect_len-1) // 2
    h1 = rect_len - h0
    for i in range(num_iter):
        # accumulates values under regular part of window (without tails!)
        accu = 0.0
        # phase a: window center still outside array
        # accumulate first h0 elements
        for k in range(-h0, 0):
            accu += in_arr[k+h0]
        # phase b: window center inside array but window does not cover array completely
        # accumulate remaining rect_len elements and write their value into array
        for k in range(0, h1):
            accu += in_arr[k+h0]
            h_arr[k] = accu
        # phase c: window completely contained in array
        # add difference of border elements and write value into array
        for k in range(h1, arr_len-h0):
            accu += (in_arr[k+h0] - in_arr[k-h1])
            h_arr[k] = accu
        # phase d (mirroring phase b): window center still inside array but window does not cover array completely
        # de-accumulate elements and write value into array
        for k in range(arr_len-h0, arr_len):
            accu -= in_arr[k-h1]
            h_arr[k] = accu
        # phase e (mirroring phase a): window center left array
        # unnecessary since value is not written

        # h_arr contains convolution result of this pass
        # swap arrays and start over next convolution
        h = in_arr
        in_arr = h_arr
        h_arr = h
    
    return in_arr


# -----------------------------------------------------------------------------

def get_half_kernel_size(sigma, step, num_iter):
    """ Computes the half kernel size T for the convolution algorithm. Version for scalar arguments. """
    return _get_half_kernel_size(_to_np(sigma), _to_np(step), num_iter)[0]


@njit
def _get_half_kernel_size(sigma, step, num_iter):
    """ Computes the half kernel size T for the convolution algorithm. Version for numpy array arguments. """
    return (np.sqrt(3.0/num_iter)*sigma/step + 0.5).astype(np.int32)


def get_sigma_effective(sigma, step, num_iter):
    """
    Computes the effective variance of the `num_iter`-fold convolved rect-kernel
    of length 2*T+1. Version for scalar arguments.
    """
    return _get_sigma_effective(_to_np(sigma), _to_np(step), num_iter)[0]


@njit
def _get_sigma_effective(sigma, step, num_iter):
    """
    Computes the effective variance of the `num_iter`-fold convolved rect-kernel
    of length 2*T+1. Version for numpy array arguments.
    """
    half_kernel_size = _get_half_kernel_size(sigma, step, num_iter)
    return np.sqrt(num_iter / 3.0 * half_kernel_size*(half_kernel_size+1)) * step
    

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

@njit
def _interpolate_radius(pts, val, sigma, x0, step, size, max_dist_weight, min_weight):
    """ 
    Implements the radius algorithm to compute the Barnes interpolation.
    Note that this algorithm only works for 2D and scalar sigma value.
    """
    offset = _normalize_values(val)

    # the grid field to store the convolved values and weights - reverse grid dimensions
    rsize = size[::-1]
    grid_value = np.zeros(rsize, dtype=np.float64)

    # construct kd-tree 'instance' with given points
    kd_tree = kdtree.create_kdtree(pts)
    
    # create kd-tree search 'instance'
    search_radius = sqrt(-2.0*log(min_weight)) * sigma
    kd_radius_search = kdtree.prepare_search(search_radius, *kd_tree)
    # extract array indices and their distances from returned tuple
    res_index, res_sqr_dist, _, _, _ = kd_radius_search
    
    scale = 2*sigma**2
    c = np.empty(2, dtype=np.float64)
    for j in range(size[1]):
        # compute y-coordinate of grid point
        c[1] = x0[1] + j*step[1]
        for i in range(size[0]):
            # compute x-coordinate of grid point
            c[0] = x0[0] + i*step[0]
            
            # loop over all observation points and compute numerator and denominator of equ. (1)
            weighted_sum = 0.0
            weight_total = 0.0
            kdtree.radius_search(c, *kd_radius_search)
            for k in range(res_index[-1]):
                weight = exp(-res_sqr_dist[k]/scale)
                weighted_sum += weight*val[res_index[k]]
                weight_total += weight
                
            # set grid points with greater distance than max_dist*sigma to NaN, i.e.
            #   points with weight < max_dist_weight
            if weight_total >= max_dist_weight:
                grid_value[j,i] = weighted_sum / weight_total + offset
            else:
                grid_value[j,i] = np.nan
            
    return grid_value


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

@njit
def _interpolate_naive(pts, val, sigma, x0, step, size):
    """
    Implements the naive algorithm A to compute the Barnes interpolation.
    """
    offset = _normalize_values(val)

    # the grid field to store the interpolated values - reverse grid dimensions
    rsize = size[::-1]
    grid_values = np.zeros(rsize, dtype=np.float64)

    scale = 2*sigma**2

    dim = len(size)
    if dim == 1:
        ptsx = pts[:,0]
        for i in range(size[0]):
            # compute x-coordinate of grid point
            xc = x0[0] + i*step[0]

            # use numpy to directly compute numerator and denominator of equ. (1)
            sqr_dist = (ptsx-xc)**2/scale[0]
            weight = np.exp(-sqr_dist)
            weighted_sum = np.dot(weight, val)
            weight_total = np.sum(weight)

            if weight_total > 0.0:
                grid_values[i] = weighted_sum / weight_total + offset
            else:
                grid_values[i] = np.nan

    elif dim == 2:
        ptsx = pts[:,0]
        ptsy = pts[:,1]
        for j in range(size[1]):
            # compute y-coordinate of grid point
            yc = x0[1] + j*step[1]
            for i in range(size[0]):
                # compute x-coordinate of grid point
                xc = x0[0] + i*step[0]

                # use numpy to directly compute numerator and denominator of equ. (1)
                sqr_dist = (ptsx-xc)**2/scale[0] + (ptsy-yc)**2/scale[1]
                weight = np.exp(-sqr_dist)
                weighted_sum = np.dot(weight, val)
                weight_total = np.sum(weight)

                if weight_total > 0.0:
                    grid_values[j,i] = weighted_sum / weight_total + offset
                else:
                    grid_values[j,i] = np.nan

    elif dim == 3:
        ptsx = pts[:,0]
        ptsy = pts[:,1]
        ptsz = pts[:,2]
        for k in range(size[2]):
            # compute z-coordinate of grid-point
            zc = x0[2] + k*step[2]
            for j in range(size[1]):
                # compute y-coordinate of grid point
                yc = x0[1] + j*step[1]
                for i in range(size[0]):
                    # compute x-coordinate of grid point
                    xc = x0[0] + i*step[0]

                    # use numpy to directly compute numerator and denominator of equ. (1)
                    sqr_dist = (ptsx-xc)**2/scale[0] + (ptsy-yc)**2/scale[1] + (ptsz-zc)**2/scale[2]
                    weight = np.exp(-sqr_dist)
                    weighted_sum = np.dot(weight, val)
                    weight_total = np.sum(weight)

                    if weight_total > 0.0:
                        grid_values[k,j,i] = weighted_sum / weight_total + offset
                    else:
                        grid_values[k,j,i] = np.nan
            
    return grid_values
