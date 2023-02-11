# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Invokes the 'optimized convolution' Barnes interpolation algorithm B on sample points
that are previously mapped to a Lambert conformal system and plots the resulting
field as isoline visualization on a geography with the same Lambert conformal
coordinate system.

NOTE
====
The execution time of this program takes around 30 seconds.

Created on Sat Jun  4 16:01:09 2022
@author: Bruno Zürcher
"""

import reader
import plotmap

import numpy as np

from fastbarnes import interpolationS2

###############################################################################

# the interpolation method
method = "optimized_convolution_S2"

# one of [ 0.25, 0.5, 1.0, 2.0, 4.0 ]
sigma = 1.0

# one of [ 54, 218, 872, 3490 ]
num_points = 3490

# one of [ 4.0, 8.0, 16.0, 32.0, 64.0 ]
resolution = 32.0

# applies only to 'convolution' interpolations: one of [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50 ]
num_iter = 4

###############################################################################

# definition of grid
step = 1.0 / resolution
x0 = np.asarray([-26.0+step, 34.5], dtype=np.float64)
size = (int(37.5/step), int(75.0/step))

# the used grid in Lambert coordinate space
lam_x0 = np.asarray([-32.0, -2.0])
lam_size = (int(44.0/step), int(64.0/step))


# read sample data from file
obs_pts, obs_values = reader.read_csv_array('../../input/obs/PressQFF_202007271200_' \
    + str(num_points) + '.csv')


# compute Barnes interpolation
res_field = interpolationS2.barnes_S2(obs_pts, obs_values, sigma, x0, step, size,
    method=method, num_iter=num_iter, resample=False)

# the underlying *fixed* Lambert projection
lambert_proj = interpolationS2.get_lambert_proj()

# display isoline plot of interpolation
plotmap.plot_Lambert_map(res_field, lam_x0, step, lam_size, lambert_proj,
    scatter_pts=obs_pts, alpha_channel=True)
