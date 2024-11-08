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

Created on Sat May 14 2022, 20:49:17
@author: Bruno Zürcher
"""

from math import exp, pi
import numpy as np

from numba import njit

from fastbarnes import interpolation
from fastbarnes.util import lambert_conformal


###############################################################################

def barnes_S2(pts, val, sigma, x0, step, size, method='optimized_convolution', num_iter=4, max_dist=3.5, resample=True):
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
        implementation (algorithm A from the paper) with an algorithmic complexity
        of O(N x W x H).
        The choice 'optimized_convolution_S2' implements the optimized algorithm B
        specified in the paper by appending tail values to the rectangular kernel.
        The latter algorithm has a reduced complexity of O(N + W x H).
        The default is 'optimized_convolution_S2'.
    num_iter : int, optional
        The number of performed self-convolutions of the underlying rect-kernel.
        Applies only if method is 'optimized_convolution_S2'.
        The default is 4.
    max_dist : float, optional
        The maximum distance between a grid point and the next sample point for which
        the Barnes interpolation is still calculated. Specified in sigma distances.
        The default is 3.5, i.e. the maximum distance is 3.5 * sigma.
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

    if method == 'optimized_convolution_S2':
        return _interpolate_opt_convol_S2(pts, val, sigma, x0, step, size, num_iter, max_dist_weight, resample)
        
    elif method == 'naive_S2':
        return _interpolate_naive_S2(pts, val, sigma, x0, step, size)
        
    else:
        raise RuntimeError("encountered invalid Barnes interpolation method: " + str(method))
    

# -----------------------------------------------------------------------------

@njit
def _interpolate_opt_convol_S2(pts, val, sigma, x0, step, size, num_iter, max_dist_weight, resample):
    """ 
    Implements the optimized convolution algorithm B for the unit sphere S^2.
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
    res1 = interpolate_opt_convol_S2_part1(pts, val, sigma, x0, step, size, num_iter, max_dist_weight)
    
    # the resampling part that performs back-projection from Lambert to lonlat space
    if resample:
        return interpolate_opt_convol_S2_part2(*res1)
    else:
        return res1[0]


@njit
def interpolate_opt_convol_S2_part1(pts, val, sigma, x0, step, size, num_iter, max_dist_weight):
    """ The convolution part of _interpolate_opt_convol_S2(), allowing to measure split times. """
    # the used Lambert projection
    lambert_proj = get_lambert_proj()
    
    # the *fixed* grid in Lambert coordinate space
    lam_x0 = np.asarray([-32.0, -2.0])
    lam_size = (int(64.0/step[0]), int(44.0/step[1]))
    
    # map lonlat sample point coordinates to Lambert coordinate space
    lam_pts = lambert_conformal.to_map(pts, pts.copy(), *lambert_proj)
    
    # call ordinary 'optimized_convolution' algorithm
    lam_field = interpolation._interpolate_opt_convol(lam_pts, val, sigma, lam_x0, step, lam_size, num_iter, max_dist_weight)

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
    """ Resamples the Lambert grid field to the specified lonlat grid. """
    # x-coordinate in lon-lat grid is constant over all grid lines
    geox = np.empty(size[0], dtype=np.float64)
    for i in range(size[0]):
        geox[i] = x0[0] + i*step[0]
        
    # memory for coordinates in Lambert space
    mapx = np.empty(size[0], dtype=np.float64)
    mapy = np.empty(size[0], dtype=np.float64)
    
    # memory for the corresponding Lambert grid indices 
    indx = np.empty(size[0], dtype=np.int32)
    indy = np.empty(size[0], dtype=np.int32)
    
    # memory for the resulting field in lonlat space
    rsize = size[::-1]
    res_field = np.empty(rsize, dtype=np.float32)
    
    # for each line in lonlat grid 
    for j in range(size[1]):
        # compute corresponding locations in Lambert space
        lambert_conformal.to_map2(geox, j*step[1] + x0[1], mapx, mapy, center_lon, n, n_inv, F, rho0)
        # compute corresponding Lambert grid indices
        mapx -= lam_x0[0]
        mapx /= step[0]
        mapy -= lam_x0[1]
        mapy /= step[1]
        # the corresponding 'i,j'-integer indices of the lower left grid point
        indx[:] = mapx.astype(np.int32)
        indy[:] = mapy.astype(np.int32)
        # and compute bilinear weights
        mapx -= indx    # contains now the weights
        mapy -= indy    # contains now the weights
        
        # compute bilinear interpolation of the 4 neighboring grid point values 
        for i in range(size[0]):
            res_field[j,i] = (1.0-mapy[i])*(1.0-mapx[i])*lam_field[indy[i],indx[i]] + \
                mapy[i]*(1.0-mapx[i])*lam_field[indy[i]+1,indx[i]] + \
                mapy[i]*mapx[i]*lam_field[indy[i]+1,indx[i]+1] + \
                (1.0-mapy[i])*mapx[i]*lam_field[indy[i],indx[i]+1]
        
    return res_field
    

# -----------------------------------------------------------------------------

@njit
def _interpolate_naive_S2(pts, val, sigma, x0, step, size):
    """ Implements the naive Barnes interpolation algorithm A for the unit sphere S^2. """
    offset = interpolation._normalize_values(val)

    # the grid field to store the interpolated values - reverse grid dimensions
    rsize = size[::-1]
    grid_val = np.zeros(rsize, dtype=np.float64)

    scale = 2*sigma**2
    for j in range(size[1]):
        # compute y-coordinate of grid point
        yc = x0[1] + j*step[1]
        for i in range(size[0]):
            # compute x-coordinate of grid point
            xc = x0[0] + i*step[0]

            # use numpy to directly compute numerator and denominator of equ. (1)
            dist = _dist_S2(xc, yc, pts[:,0], pts[:,1])
            weight = np.exp(-dist*dist/scale[0])        # assuming scale is equal in x and y direction
            weighted_sum = np.dot(weight, val)
            weight_total = np.sum(weight)

            if weight_total > 0.0:
                grid_val[j,i] = weighted_sum / weight_total + offset
            else:
                grid_val[j,i] = np.nan

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
