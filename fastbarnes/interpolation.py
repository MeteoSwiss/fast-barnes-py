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

Created on Sat May 14 13:10:47 2022
@author: Bruno Zürcher
"""

from math import sqrt, exp, log, pi
import numpy as np

from numba import njit

from fastbarnes.util import kdtree


###############################################################################

def barnes(pts, val, sigma, x0, step, size, method='optimized_convolution',
    num_iter=4, min_weight=0.001):
    """
    Computes the Barnes interpolation for observation values `val` taken at sample
    points `pts` using Gaussian weights for the width parameter `sigma`.
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
    method : {'optimized_convolution', 'convolution', 'radius', 'naive'}
        Designates the Barnes interpolation method to be used. The possible
        implementations that can be chosen are 'naive' for the straightforward
        implementation (algorithm A from paper), 'radius' to consider only sample
        points within a specific radius of influence, both with an algorithmic
        complexity of O(N x W x H).
        The choice 'convolution' implements algorithm B specified in the paper
        and 'optimized_convolution' is its optimization by appending tail values
        to the rectangular kernel. The latter two algorithms reduce the complexity
        down to O(N + W x H).
        The default is 'optimized_convolution'.
    num_iter : int, optional
        The number of performed self-convolutions of the underlying rect-kernel.
        Applies only if method is 'optimized_convolution' or 'convolution'.
        The default is 4.
    min_weight : float, optional
        Choose radius of influence such that Gaussian weight of considered sample
        points is greater than `min_weight`.
        Applies only if method is 'radius'. Recommended values are 0.001 and less.
        The default is 0.001, which corresponds to a radius of 3.717 * sigma.

    Returns
    -------
    numpy ndarray
        A 2-dimensional array containing the resulting field of the performed
        Barnes interpolation.
    """

    if method == 'optimized_convolution':
        return _interpolate_opt_convol(pts, val, sigma, x0, step, size, num_iter)
        
    elif method == 'convolution':
        return _interpolate_convol(pts, val, sigma, x0, step, size, num_iter)
    
    elif method == 'radius':
        return _interpolate_radius(pts, val, sigma, x0, step, size, min_weight)
        
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

@njit
def _inject_data(vg, wg, pts, val, x0, step, size):
    """
    Injects the observations values and weights, respectively, into the
    corresponding fields as described by algorithm B.1.
    """
    for k in range(len(pts)):
        xc = (pts[k,0]-x0[0]) / step
        yc = (pts[k,1]-x0[1]) / step
        if (xc < 0.0 or yc < 0.0 or xc >= size[1]-1 or yc >= size[0]-1):
            continue
        xi = int(xc)
        yi = int(yc)
        xw = xc - xi
        yw = yc - yi
        
        w = (1.0-xw)*(1.0-yw)
        vg[yi, xi] += w*val[k]
        wg[yi, xi] += w

        w =  xw*(1.0-yw)
        vg[yi, xi+1] +=w*val[k]
        wg[yi, xi+1] += w
        
        w = xw*yw
        vg[yi+1, xi+1] += w*val[k]
        wg[yi+1, xi+1] += w
        
        w = (1.0-xw)*yw
        vg[yi+1, xi] += w*val[k]
        wg[yi+1, xi] += w


# -----------------------------------------------------------------------------

@njit
def _interpolate_opt_convol(pts, val, sigma, x0, step, size, num_iter):
    """ 
    Implements algorithm B presented in section 4 of the paper but optimized for
    a rectangular window with a tail value alpha.
    """
    offset = _normalize_values(val)

    # the grid fields to store the convolved values and weights
    vg = np.zeros(size, dtype=np.float64)
    wg = np.zeros(size, dtype=np.float64)
    
    # inject obs values into grid
    _inject_data(vg, wg, pts, val, x0, step, size)
        
    # prepare convolution
    half_kernel_size = get_half_kernel_size_opt(sigma, step, num_iter)
    kernel_size = 2*half_kernel_size + 1
        
    tail_value = get_tail_value(sigma, step, num_iter)
    
    # execute algorithm B
    # convolve rows in x-direction
    h_arr = np.empty(size[1], dtype=np.float64)
    for j in range(size[0]):
        # convolve row values
        vg[j,:] = _accumulate_tail_array(vg[j,:].copy(), h_arr, size[1], kernel_size, num_iter, tail_value)
            
        # convolve row weights
        wg[j,:] = _accumulate_tail_array(wg[j,:].copy(), h_arr, size[1], kernel_size, num_iter, tail_value)
        
    # convolve columns in y- direction
    h_arr = np.empty(size[0], dtype=np.float64)
    for i in range(size[1]):
        # convolve column values
        vg[:,i] = _accumulate_tail_array(vg[:,i].copy(), h_arr, size[0], kernel_size, num_iter, tail_value)
        
        # convolve column weights
        wg[:,i] = _accumulate_tail_array(wg[:,i].copy(), h_arr, size[0], kernel_size, num_iter, tail_value)
        
    # compute limit wg array value for which weight > 0.0022, i.e. grid points with greater distance
    #   than 3.5*sigma will evaluate to NaN
    # since we dropped common factors in our computation, we have to revert their cancellation in the
    #   following computation
    factor = (kernel_size+2*tail_value) ** (2*num_iter) * (step/sigma) ** 2 / 2 / pi * 0.0022

    # set smaller weights to NaN with overall effect that corresponding quotient is NaN, too
    for j in range(size[0]):
        for i in range(size[1]):
            if wg[j,i] < factor: wg[j,i] = np.nan

    # yet to be considered:
    # - add offset again to resulting quotient
    # - and apply quantization operation:
    #   here by casting double to float and thus drop around 29 least significant bits
    return (vg / wg + offset).astype(np.float32)

    
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

    
@njit
def get_half_kernel_size_opt(sigma, step, num_iter):
    """ Computes the half kernel size T for the optimized convolution algorithm. """
    s = sigma / step
    return int((sqrt(1.0+12*s*s/num_iter) - 1) / 2)


@njit
def get_tail_value(sigma, step, num_iter):
    """ Computes the tail value alpha for the optimized convolution algorithm. """
    half_kernel_size = get_half_kernel_size_opt(sigma, step, num_iter)
    kernel_size = 2*half_kernel_size + 1

    sigma_rect_sqr = (half_kernel_size+1)*half_kernel_size/3.0*step**2
    # slightly rearranged expression from equ. (12)
    return  0.5*kernel_size*(sigma**2/num_iter - sigma_rect_sqr) \
        / (((half_kernel_size+1)*step)**2 - sigma**2/num_iter)


# -----------------------------------------------------------------------------

@njit
def _interpolate_convol(pts, val, sigma, x0, step, size, num_iter):
    """ 
    Implements algorithm B presented in section 4 of the paper.
    """
    offset = _normalize_values(val)

    # the grid fields to store the convolved values and weights
    vg = np.zeros(size, dtype=np.float64)
    wg = np.zeros(size, dtype=np.float64)
    
    # inject obs values into grid
    _inject_data(vg, wg, pts, val, x0, step, size)
        
    # prepare convolution
    half_kernel_size = get_half_kernel_size(sigma, step, num_iter)
    kernel_size = 2*half_kernel_size + 1
        
    # execute algorithm B
    # convolve rows in x-direction
    h_arr = np.empty(size[1], dtype=np.float64)
    for j in range(size[0]):
        # convolve row values
        vg[j,:] = _accumulate_array(vg[j,:].copy(), h_arr, size[1], kernel_size, num_iter)
            
        # convolve row weights
        wg[j,:] = _accumulate_array(wg[j,:].copy(), h_arr, size[1], kernel_size, num_iter)
        
    # convolve columns in y- direction
    h_arr = np.empty(size[0], dtype=np.float64)
    for i in range(size[1]):
        # convolve column values
        vg[:,i] = _accumulate_array(vg[:,i].copy(), h_arr, size[0], kernel_size, num_iter)
        
        # convolve column weights
        wg[:,i] = _accumulate_array(wg[:,i].copy(), h_arr, size[0], kernel_size, num_iter)
        
    
    # compute limit wg array value for which weight > 0.0022, i.e. grid points with greater distance
    #   than 3.5*sigma will evaluate to NaN
    # since we dropped common factors in our computation, we have to revert their cancellation in the
    #   following computation
    sigma_eff = get_sigma_effective(sigma, step, num_iter)
    factor = float(kernel_size) ** (2*num_iter) * (step/sigma_eff) ** 2 / 2 / pi * 0.0022
    
    # set smaller weights to NaN with overall effect that corresponding quotient is NaN, too
    for j in range(size[0]):
        for i in range(size[1]):
            if wg[j,i] < factor: wg[j,i] = np.nan
    
    # yet to be considered:
    # - add offset again to resulting quotient
    # - and apply quantization operation:
    #   here by temporary casting double to float and thus drop around 29 least significant bits
    return (vg / wg + offset).astype(np.float32)


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


@njit
def get_half_kernel_size(sigma, step, num_iter):
    """ Computes the half kernel size T for the convolution algorithm. """
    return int(sqrt(3.0/num_iter)*sigma/step + 0.5)


@njit
def get_sigma_effective(sigma, step, num_iter):
    """
    Computes the effective variance of the `num_iter`-fold convolved rect-kernel
    of length 2*T+1.
    """
    half_kernel_size = get_half_kernel_size(sigma, step, num_iter)
    return sqrt(num_iter / 3.0 * half_kernel_size*(half_kernel_size+1)) * step
    

# -----------------------------------------------------------------------------

@njit
def _interpolate_radius(pts, val, sigma, x0, step, size, min_weight):
    """ 
    Implements the radius algorithm to compute the Barnes interpolation.
    """
    offset = _normalize_values(val)

    grid_value = np.zeros(size, dtype=np.float64)

    # construct kd-tree 'instance' with given points
    kd_tree = kdtree.create_kdtree(pts)
    
    # create kd-tree search 'instance'
    search_radius = sqrt(-2.0*log(min_weight)) * sigma
    kd_radius_search = kdtree.prepare_search(search_radius, *kd_tree)
    # extract array indices and their distances from returned tuple
    res_index, res_sqr_dist, _, _, _ = kd_radius_search
    
    scale = 2*sigma**2
    c = np.empty(2, dtype=np.float64)
    for j in range(size[0]):
        # compute y-coordinate of grid point
        c[1] = x0[1] + j*step
        for i in range(size[1]):
            # compute x-coordinate of grid point
            c[0] = x0[0] + i*step
            
            # loop over all observation points and compute numerator and denominator of equ. (1)
            weighted_sum = 0.0
            weight_total = 0.0
            kdtree.radius_search(c, *kd_radius_search)
            for k in range(res_index[-1]):
                weight = exp(-res_sqr_dist[k]/scale)
                weighted_sum += weight*val[res_index[k]]
                weight_total += weight
                
            # set grid points with greater distance than 3.5*sigma to NaN, i.e.
            #   points with weight < 0.0022, 
            if weight_total >= 0.0022:
                grid_value[j,i] = weighted_sum / weight_total + offset
            else:
                grid_value[j,i] = np.nan
            
    return grid_value


# -----------------------------------------------------------------------------

@njit
def _interpolate_naive(pts, val, sigma, x0, step, size):
    """ Implements the naive algorithm A to compute the Barnes interpolation. """
    offset = _normalize_values(val)
    
    grid_value = np.zeros(size, dtype=np.float64)
    
    scale = 2*sigma**2
    for j in range(size[0]):
        # compute y-coordinate of grid point
        yc = x0[1] + j*step
        for i in range(size[1]):
            # compute x-coordinate of grid point
            xc = x0[0] + i*step
            
            # use numpy to directly compute numerator and denominator of equ. (1)
            sqr_dist = (pts[:,0]-xc)**2 + (pts[:,1]-yc)**2
            weight = np.exp(-sqr_dist/scale)
            weighted_sum = np.dot(weight, val)
            weight_total = np.sum(weight)
            
            if weight_total > 0.0:
                grid_value[j,i] = weighted_sum / weight_total + offset
            else:
                grid_value[j,i] = np.nan
            
    return grid_value
