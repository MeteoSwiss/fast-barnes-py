# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Runs the 'convolution' and 'optimized_convolution' Barnes interpolation algorithms
for a set of iteration numbers and compiles the plots of a West European subdomain
in a composite image.

NOTE
====
The execution time of this program takes around 2 minutes.

Created on Sun Jun 5 2022, 19:19:44
@author: Bruno Zürcher
"""

import reader
import plotmap

import numpy as np

from PIL import Image, ImageFont, ImageDraw

from fastbarnes import interpolation


# the test parameters #########################################################

# the number of samples
num_points = 3490

# the result data resolution
resolution = 32.0

# the Gaussian width parameter
sigma = 1.0


# other parameters, not necessarily to modify #################################

# the different interpolation methods
methods_set = [ 'convolution', 'optimized_convolution' ]

# the number of iterations to plot 
num_iter_set = [ 1, 2, 4, 6, 20, 50 ]


###############################################################################

# cut region in map images
x00 = 370
y00 = 404
w = 143
h = 239

# specification of region in destination image
u0 = 10
v0 = 32
W = w + 10
H = h + 44


destIm = Image.new("RGB", (u0+len(num_iter_set)*W, v0+2*H-6), "white")

font = ImageFont.truetype('arial.ttf', 20)

editIm = ImageDraw.Draw(destIm)


# definition of grid
step = 1.0 / resolution
x0 = np.asarray([-26.0+step, 34.5], dtype=np.float64)
size = (int(75.0/step), int(37.5/step))


# read sample data from file
obs_pts, obs_values = reader.read_csv_array('input/PressQFF_202007271200_' + str(num_points) + '.csv')


# create interpolation images
num = 0
for num_iter in num_iter_set:
    # compute convolution Barnes interpolation image
    res_field = interpolation.barnes(obs_pts, obs_values.copy(), sigma, x0, step, size,
        method='convolution', num_iter=num_iter)
    image = plotmap.image_lat_lon_map(res_field, x0, step, size, scatter_pts=obs_pts, line_labels=False)

    # image.save('../temp/test.png')

    region = image.crop((x00, y00, x00+w, y00+h))
    destIm.paste('black', (u0+num*W-1, v0-1, u0+num*W+w+1, v0+h+1))
    destIm.paste(region, (u0+num*W, v0, u0+num*W+w, v0+h))
    
    # compute optimized convolution Barnes interpolation image
    res_field = interpolation.barnes(obs_pts, obs_values.copy(), sigma, x0, step, size,
        method='optimized_convolution', num_iter=num_iter)
    image = plotmap.image_lat_lon_map(res_field, x0, step, size, scatter_pts=obs_pts, line_labels=False)

    region = image.crop((x00, y00, x00+w, y00+h))
    destIm.paste('black', (u0+num*W-1, v0+H-1, u0+num*W+w+1, v0+h+H+1))
    destIm.paste(region, (u0+num*W, v0+H, u0+num*W+w, v0+h+H))

    # add label text to image columns
    editIm.text((w/2+num*W-9,4), 'n = '+str(num_iter), 'black', font=font)
        
    num += 1

editIm.text((len(num_iter_set)//2*W-144, H-7), '(a) original convolution algorithm', 'black', font=font)
editIm.text((len(num_iter_set)//2*W-144, 2*H-7), '(b) optimized convolution algorithm', 'black', font=font)

destIm.show()

