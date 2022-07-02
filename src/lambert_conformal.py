# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Implements Lambert conformal conic projection.

Uses Numba JIT compiler and numpy arrays to achieve the best performance with the
downside that Python class formalism cannot be used and has to be emulated using
data containers and 'external' functions acting on them.

Assuming the lon/lat-coordinates of the points to be transformed are given by the
array `pts`, the corresponding Lambert map coordinates can be computed by:


# create Lambert projection 'instance'
lambert_proj = lambert_conformal.create_proj(11.5, 34.5, 42.5, 65.5)

# map lonlat sample point coordinates to Lambert coordinate space
pts = np.asarray([[8.55, 47.37], [13.41, 52.52], ...])
lam_pts = lambert_conformal.to_map(pts, pts.copy(), *lambert_proj)


The reverse transform can be invoked by calling lambert_conformal.to_geo().

Refer to Snyder J. (1987), Map Projections: A Working Manual, US Geological
Survey Professional Paper 1395, US Government Printing Office, Washington.    


Created on Sat Jun  4 13:18:04 2022
@author: Bruno Zürcher
"""

from math import log, cos, sin, tan, pi
import numpy as np

from numba import njit

###############################################################################


RAD_PER_DEGREE = pi / 180.0
HALF_RAD_PER_DEGREE = RAD_PER_DEGREE / 2.0


@njit
def create_proj(center_lon, center_lat, lat1, lat2):
    """
    Creates a Lambert conformal projection 'instance' using the specified parameters.
    
    In absence of a class concept supported by Numba, the projection 'instance'
    merely consists of a tuple containing the relevant Lambert projection constants.

    Parameters
    ----------
    center_lon : float
        The center meridian, which points in the upward direction of the map [°E].
    center_lat : float
        The center parallel, which defines together with the center meridian
        the origin of the map [°N].
    lat1 : float
        The latitude of the first (northern) standard parallel [°N].
    lat2 : float
        The latitude of the second (southern) standard parallel [°S].

    Returns
    -------
    center_lon : float
        The center meridian, as above.
    n : float
        The cone constant, i.e. the ratio of the angle between meridians to the true
        angle, as described in Snider.
    n_inv : float
        The inverse of n, as described in Snider.
    F : float
        A constant used for mapping, as described in Snider.
    rho0 : float
        Unscaled distance from the cone tip to the first standard parallel, as
        described in Snider.
    """

    if lat1 != lat2:
        n = log(cos(lat1 *RAD_PER_DEGREE) / cos(lat2 *RAD_PER_DEGREE)) / log(tan((90.0+ lat2)*HALF_RAD_PER_DEGREE) / tan((90.0+ lat1)*HALF_RAD_PER_DEGREE))
    else:
        n = sin(lat1 * RAD_PER_DEGREE)
        
    n_inv = 1.0 / n
    F = cos(lat1 * RAD_PER_DEGREE) * tan((90.0+lat1)*HALF_RAD_PER_DEGREE) ** n / n
    rho0 = F / tan((90.0+center_lat)*HALF_RAD_PER_DEGREE) ** n
    
    return (center_lon, n, n_inv, F, rho0)


# -----------------------------------------------------------------------------

@njit
def get_scale(lat, center_lon, n, n_inv, F, rho0):
    """
    Returns the local scale factor for the specified latitude.
    This quantity specifies the degree of length distortion compared to that at
    the standard latitudes of this Lambert conformal projection instance, where
    it is by definition 1.0.
    """
    return F * n / (np.cos(lat*RAD_PER_DEGREE) * np.power(np.tan((90.0+lat)*HALF_RAD_PER_DEGREE), n))


# -----------------------------------------------------------------------------

@njit
def to_map(geoc, mapc, center_lon, n, n_inv, F, rho0):
    """
    Maps the geographic coordinates given by the numpy array `geoc` onto the Lambert
    map given by the tuple `(center_lon, n, n_inv, F, rho0)` and stores the result
    in the preallocated numpy array `mapc`.
    """
    rho = F / np.power(np.tan((90.0+geoc[:,1])*HALF_RAD_PER_DEGREE), n)
    arg = n * (geoc[:,0] - center_lon) * RAD_PER_DEGREE
    mapc[:,0] = rho * np.sin(arg) / RAD_PER_DEGREE
    mapc[:,1] = (rho0 - rho*np.cos(arg)) / RAD_PER_DEGREE
    return mapc


@njit
def to_map2(geox, geoy, mapx, mapy, center_lon, n, n_inv, F, rho0):
    """
    Maps the geographic coordinates given by the separated numpy arrays `geox` and `geoy`
    onto the Lambert map given by the tuple `(center_lon, n, n_inv, F, rho0)` and
    stores the result in the preallocated numpy arrays `mapx` and `mapy`.
    """
    rho = F / np.power(np.tan((90.0+geoy)*HALF_RAD_PER_DEGREE), n)
    arg = n * (geox - center_lon) * RAD_PER_DEGREE
    mapx[:] = rho * np.sin(arg) / RAD_PER_DEGREE
    mapy[:] = (rho0 - rho*np.cos(arg)) / RAD_PER_DEGREE
    
#------------------------------------------------------------------------------

@njit
def to_geo(mapc, geoc, center_lon, n, n_inv, F, rho0):
    """
    Maps the Lambert map coordinates given by the numpy array `mapc` to the
    geographic coordinate system and stores the result in the preallocated numpy
    array `geoc`. The Lambert projection is given by the tuple
    `(center_lon, n, n_inv, F, rho0)`.
    """
    x = mapc[:,0] * RAD_PER_DEGREE
    y = mapc[:,1] * RAD_PER_DEGREE
    arg = rho0 - y
    rho = np.sqrt(x**2 + arg**2)
    if (n < 0.0):
        rho = np.negative(rho)
    theta = np.arctan2(x, arg)
    geoc[:,1] = np.arctan(np.power(F/rho, n_inv)) / HALF_RAD_PER_DEGREE - 90.0
    geoc[:,0] = center_lon + theta / n / RAD_PER_DEGREE
    return geoc
    

@njit
def to_geo2(mapx, mapy, geox, geoy, center_lon, n, n_inv, F, rho0):
    """
    Maps the Lambert map coordinates given by the numpy arrays `mapx` and `mapy` to
    the geographic coordinate system and stores the result in the preallocated numpy
    arrays `geox` and `geoy`. The Lambert projection is given by the tuple
    `(center_lon, n, n_inv, F, rho0)`.
    """
    x = mapx * RAD_PER_DEGREE
    arg = rho0 - mapy * RAD_PER_DEGREE
    rho = np.sqrt(x**2 + arg**2)
    if (n < 0.0):
        rho = np.negative(rho)
    theta = np.arctan2(x, arg)
    geoy[:] = np.arctan(np.power(F/rho, n_inv)) / HALF_RAD_PER_DEGREE - 90.0
    geox[:] = center_lon + theta / n / RAD_PER_DEGREE    
