# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Computes the 'convolution' Barnes interpolation algorithm B with a constellation of
QFF values over Iceland that trigger an artifact when plotted with an isoline
visualization.
The plot shows the situation, if no quantization takes place.

Created on Sun Jun 5 2022, 19:51:02
@author: Bruno Zürcher
"""

import quantinterpolation
import reader
import plotmap

import numpy as np

###############################################################################

# the Barnes interpolation method
method = 'convolution'

# the Gaussian width parameter
sigma = 0.75

# the grid resolution
resolution = 32.0

# the number of iterations
num_iter = 4

# the number of quantization bits: one of [ 0, 3, 6 ]
quant_bits = 0

###############################################################################

# definition of grid
step = 1.0 / resolution
x0 = np.asarray([-26.0+step, 34.5], dtype=np.float64)
size = (int(75.0/step), int(37.5/step))


# read sample data from file
obs_pts, obs_values = reader.read_csv_array('input/Iceland_PressOFF_constellation.csv')


# compute Barnes interpolation with special quantization implementation
res_field = quantinterpolation.barnes(obs_pts, obs_values, sigma, x0, step, size,
    quant_bits, num_iter)

# display isoline plot of interpolation
plotmap.plot_Iceland_map(res_field, x0, step, size, scatter_pts=obs_pts,
    alpha_channel=True, line_labels=False)
