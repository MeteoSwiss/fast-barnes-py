# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Module that provides two different Barnes interpolation algorithms acting on
the unit sphere S^2 and thus using the spherical distance metric.
To attain competitive performance, the code is written using Numba's
just-in-time compiler and thus has to use the respective programming idiom,
which is sometimes not straightforward to read at a first glance. Allocated
memory is as far as possible reused in order to reduce the workload imposed
on the garbage collector.

Created on Sat May 14 20:49:17 2022
@author: Bruno Zürcher
"""

from math import pi
import numpy as np

from numba import njit

from fastbarnes import interpolation
from fastbarnes.util import lambert_conformal


###############################################################################

def barnes_S2(pts, val, sigma, x0, step, size, method='optimized_convolution', num_iter=4, resample=True):
    """
    Computes the Barnes interpolation for observation values `val` taken at sample
    points `pts` using Gaussian weights for the width parameter `sigma`.
    The underlying grid is embedded on the unit sphere S^2 and thus inherits the
    spherical distance measure (taken in degrees). The grid is given by the start
    point `x0`, regular grid steps `step` and extension `size`.

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
    method : {'optimized_convolution_S2', 'naive_S2'}
        Designates the Barnes interpolation method to be used. The possible
        implementations that can be chosen are 'naive_S2' for the straightforward
        implementation with an algorithmic complexity of O(N x W x H).
        The choice 'optimized_convolution_S2' implements the optimized algorithm 4
        specified in the paper by appending tail values to the rectangular kerne
        The latter algorithm has a reduced complexity of O(N + W x H).
        The default is 'optimized_convolution_S2'.
    num_iter : int, optional
        The number of performed self-convolutions of the underlying rect-kernel.
        Applies only if method is 'optimized_convolution_S2'.
        The default is 4.
    resample : bool, optional
        Specifies whether to resample Lambert grid field to lonlat grid.
        Applies only if method is 'optimized_convolution_S2'.
        The default is True.

    Returns
    -------
    numpy ndarray
        A 2-dimensional array containing the resulting field of the performed
        Barnes interpolation.
    """    

    if method == 'optimized_convolution_S2':
        return _interpolate_opt_convol_S2(pts, val, sigma, x0, step, size, num_iter, resample)
        
    elif method == 'naive_S2':
        return _interpolate_naive_S2(pts, val, sigma, x0, step, size)
        
    else:
        raise RuntimeError("encountered invalid Barnes interpolation method: " + str(method))
    

# -----------------------------------------------------------------------------

@njit
def _interpolate_opt_convol_S2(pts, val, sigma, x0, step, size, num_iter, resample):
    """ 
    Implements the optimized convolution algorithm 4 for the unit sphere S^2.
    """
    # # the used Lambert projection
    # lambert_proj = get_lambert_proj()
    
    # # the *fixed* grid in Lambert coordinate space
    # lam_x0 = np.asarray([-32.0, -2.0])
    # lam_size = (int(44.0/step), int(64.0/step))
    
    # # map lonlat sample point coordinatess to Lambert coordinate space
    # lam_pts = lambert_conformal.to_map(pts, pts.copy(), *lambert_proj)
    
    # # call ordinary 'optimized_convolution' algorithm
    # lam_field = interpolation._interpolate_opt_convol(lam_pts, val, sigma, lam_x0, step, lam_size, num_iter)
    
    # if resample:
    #     return _resample(lam_field, lam_x0, x0, step, size, *lambert_proj)
    # else:
    #     return lam_field
    
    
    
    # split commented code above in two separately 'measurable' sub-routines
    
    # the convolution part taking place in Lambert space
    res1 = interpolate_opt_convol_S2_part1(pts, val, sigma, x0, step, size, num_iter)
    
    # the resampling part that performs back-projection from Lambert to lonlat space
    if resample:
        return interpolate_opt_convol_S2_part2(*res1)
    else:
        return res1[0]


@njit
def interpolate_opt_convol_S2_part1(pts, val, sigma, x0, step, size, num_iter):
    """ The convolution part of _interpolate_opt_convol_S2(), allowing to measure split times. """
    # the used Lambert projection
    lambert_proj = get_lambert_proj()
    
    # the *fixed* grid in Lambert coordinate space
    lam_x0 = np.asarray([-32.0, -2.0])
    lam_size = (int(44.0/step), int(64.0/step))
    
    # map lonlat sample point coordinates to Lambert coordinate space
    lam_pts = lambert_conformal.to_map(pts, pts.copy(), *lambert_proj)
    
    # call ordinary 'optimized_convolution' algorithm
    lam_field = interpolation._interpolate_opt_convol(lam_pts, val, sigma, lam_x0, step, lam_size, num_iter)

    return (lam_field, lam_x0, x0, step, size, lambert_proj)


@njit
def interpolate_opt_convol_S2_part2(lam_field, lam_x0, x0, step, size, lambert_proj):
    """ The back-projection part of _interpolate_opt_convol_S2(), allowing to measure split times. """
    return _resample(lam_field, lam_x0, x0, step, size, *lambert_proj)

    
@njit
def get_lambert_proj():
    """ Return the Lambert projection that is used for our test example. """
    return lambert_conformal.create_proj(11.5, 34.5, 42.5, 65.5)

    
@njit
def _resample(lam_field, lam_x0, x0, step, size, center_lon, n, n_inv, F, rho0):
    """ Resamples the Lambert grdi field to the specified lonlat grid. """
    # x-coordinate in lon-lat grid is constant over all grid lines
    geox = np.empty(size[1], dtype=np.float64)
    for i in range(size[1]):
        geox[i] = x0[0] + i*step
        
    # memory for coordinates in Lambert space
    mapx = np.empty(size[1], dtype=np.float64)
    mapy = np.empty(size[1], dtype=np.float64)
    
    # memory for the corresponding Lambert grid indices 
    indx = np.empty(size[1], dtype=np.int32)
    indy = np.empty(size[1], dtype=np.int32)
    
    # memory for the resulting field in lonlat space
    res_field = np.empty(size, dtype=np.float32)
    
    # for each line in lonlat grid 
    for j in range(size[0]):
        # compute corresponding locations in Lambert space
        lambert_conformal.to_map2(geox, j*step + x0[1], mapx, mapy, center_lon, n, n_inv, F, rho0)
        # compute corresponding Lambert grid indices
        mapx -= lam_x0[0]
        mapx /= step
        mapy -= lam_x0[1]
        mapy /= step
        # the corresponding 'i,j'-integer indices of the lower left grid point
        indx[:] = mapx.astype(np.int32)
        indy[:] = mapy.astype(np.int32)
        # and compute bilinear weights
        mapx -= indx    # contains now the weights
        mapy -= indy    # contains now the weights
        
        # compute bilinear interpolation of the 4 neighboring grid point values 
        for i in range(size[1]):
            res_field[j,i] = (1.0-mapy[i])*(1.0-mapx[i])*lam_field[indy[i],indx[i]] + \
                mapy[i]*(1.0-mapx[i])*lam_field[indy[i]+1,indx[i]] + \
                mapy[i]*mapx[i]*lam_field[indy[i]+1,indx[i]+1] + \
                (1.0-mapy[i])*mapx[i]*lam_field[indy[i],indx[i]+1]
        
    return res_field
    

# -----------------------------------------------------------------------------

@njit
def _interpolate_naive_S2(pts, val, sigma, x0, step, size):
    """ Implements the naive Barnes interpolation algorithm for the unit sphere S^2. """
    offset = interpolation._normalize_values(val)
    
    grid_val = np.zeros(size, dtype=np.float64)
    
    scale = 2*sigma**2
    for j in range(size[0]):
        # compute y-coordinate of grid point
        yc = x0[1] + j*step
        for i in range(size[1]):
            # compute x-coordinate of grid point
            xc = x0[0] + i*step
            
            # use numpy to directly compute numerator and denominator of equ. (1)
            dist = _dist_S2(xc, yc, pts[:,0], pts[:,1])
            weight = np.exp(-dist*dist/scale)
            weighted_sum = np.dot(weight, val)
            weight_total = np.sum(weight)
            
            if weight_total > 0.0:
                grid_val[j,i] = weighted_sum / weight_total + offset
            else:
                grid_val[j,i] = np.NaN
            
    return grid_val



RAD_PER_DEGREE = pi / 180.0


@njit
def _dist_S2(lon0, lat0, lon1, lat1):
    """ Computes spherical distance between the 2 specified points. Input and output in degrees. """
    lat0_rad = lat0 * RAD_PER_DEGREE
    lat1_rad = lat1 * RAD_PER_DEGREE
    arg = np.sin(lat0_rad)*np.sin(lat1_rad) + np.cos(lat0_rad)*np.cos(lat1_rad)*np.cos((lon1-lon0)*RAD_PER_DEGREE)
    arg[arg > 1.0] = 1.0
    return np.arccos(arg) / RAD_PER_DEGREE
