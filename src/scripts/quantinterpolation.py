# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Module that provides the same 'convolution' algorithm as in the regular interpolation
module, up to the fact that the floating point number quantization degree is exposed
in the API and thus can be freely chosen.

Created on Sun Jun 05 11:31:27 2022
@author: Bruno Zürcher
"""

from quantization import quant

from math import pi
import numpy as np

from numba import njit

import sys
sys.path.append('..')

import interpolation

###############################################################################

def barnes(pts, val, sigma, x0, step, size, quant_bits, num_iter=4):
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

    Returns
    -------
    numpy ndarray
        A 2-dimensional array containing the resulting field of the performed
        Barnes interpolation.
    """
    
    return _interpolate_quant_convol(pts, val, sigma, x0, step, size, quant_bits, num_iter)
    

#------------------------------------------------------------------------------

@njit
def _interpolate_quant_convol(pts, val, sigma, x0, step, size, quant_bits, num_iter):
    """ 
    Implements algorithm 4 presented in section 4 of the paper.
    In contrast to the _interpolate_convol() function found in ordinary interpolation
    module that applies a fixed quantization of 29 bits, this function allows the
    caller to freely choose the number of quantization bits.
    """
    offset = interpolation._normalize_values(val)

    # the grid fields to store the convolved values and weights
    vg = np.zeros(size, dtype=np.float64)
    wg = np.zeros(size, dtype=np.float64)
    
    # inject obs values into grid
    interpolation._inject_data(vg, wg, pts, val, x0, step, size)
    
    # prepare convolution
    half_kernel_size = interpolation.get_half_kernel_size(sigma, step, num_iter)
    kernel_size = 2*half_kernel_size + 1
        
    # execute algorithm 4
    # convolve rows in x-direction
    h_arr = np.empty(size[1], dtype=np.float64)
    for j in range(size[0]):
        # convolve row values
        vg[j,:] = interpolation._accumulate_array(vg[j,:].copy(), h_arr, size[1], kernel_size, num_iter)
            
        # convolve row weights
        wg[j,:] = interpolation._accumulate_array(wg[j,:].copy(), h_arr, size[1], kernel_size, num_iter)
        
    # convolve columns in y- direction
    h_arr = np.empty(size[0], dtype=np.float64)
    for i in range(size[1]):
        # convolve column values
        vg[:,i] = interpolation._accumulate_array(vg[:,i].copy(), h_arr, size[0], kernel_size, num_iter)
        
        # convolve column weights
        wg[:,i] = interpolation._accumulate_array(wg[:,i].copy(), h_arr, size[0], kernel_size, num_iter)
        
    
    # compute limit wgtArr value for which weight > 0.0022, i.e. grid points with greater distance
    #   than 3.5*sigma will evaluate to NaN
    # since we dropped common factors in our computation, we have to revert their cancellation in the
    #   following computation
    sigma_eff = interpolation.get_sigma_effective(sigma, step, num_iter)
    factor = float(kernel_size) ** (2*num_iter) * (step/sigma_eff) ** 2 / 2 / pi * 0.0022
    
    # set smaller weights to NaN with overall effect that corresponding quotient is NaN, too
    for j in range(size[0]):
        for i in range(size[1]):
            if wg[j,i] < factor: wg[j,i] = np.NaN
    
    # yet to be considered:
    # - add offset again to resulting quotient
    vg = vg / wg + offset
    
    ### HERE DIFFERS IMPLEMENTATION ###
    # - and apply quantization operation: remove specified number of quantization bits
    quant(vg, quant_bits)
    
    return vg
