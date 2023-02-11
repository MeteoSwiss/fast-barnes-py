# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Invokes the 'naive' Barnes interpolation algorithm A using great circle distances
(the metric on S^2) and plots the resulting field as isoline visualization on a
geography with a lon-lat coordinate system.

NOTE
====
The execution time of this program takes around 12 minutes.
You can reduce this time by decreasing for instance the resolution or the number
of used sample points.

Created on Sat Jan 23 17:31:00 2021
@author: Bruno Zürcher
"""

import reader
import plotmap

import numpy as np

from fastbarnes import interpolationS2

###############################################################################

# one of [ 'naive_S2', 'optimized_convolution_S2' ]
method = "naive_S2"

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


# read sample data from file
obs_pts, obs_values = reader.read_csv_array('../../input/obs/PressQFF_202007271200_' + str(num_points) + '.csv')


# compute Barnes interpolation
res_field = interpolationS2.barnes_S2(obs_pts, obs_values, sigma, x0, step, size,
    method=method, num_iter=num_iter)


# display isoline plot of interpolation
plotmap.plot_lat_lon_map(res_field, x0, step, size, scatter_pts=obs_pts, alpha_channel=True)
