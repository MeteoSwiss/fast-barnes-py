# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Runs the 'convolution' and 'optimized_convolution' Barnes interpolation algorithms
with a number of iterations that increases from 1 to 50 and computes the respective
RMSEs of a West European subdomain, when compared to the exact baseline given by the
'naive', but exact Barnes interpolation.

This script produces a graphic, in which only every second x-axis label is shown
and the labels of the y-axis are thinned out as well. Thus, the used font can be
chosen somewhat larger and is therefore better readable.

NOTE
====
The execution time of this program takes around 6 minutes.
You can reduce this time by decreasing the resolution to 16.0 or 8.0 for instance.

Created on Sun May 29 2022, 17:01:51
@author: Bruno Zürcher
"""

import reader
from rmse import rmse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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

# used for table outputs
method_short_names = [ 'conv', 'opt_conv' ]

# the number of iterations to test 
num_iter_set = range(1, 51)


# do not modify after this point ##############################################

print('Barnes Interpolation RMSE Measurements')
print('**************************************')
print()
print('number of points: ', num_points)
print('resolution:       ', resolution)
print('sigma:            ', sigma)


# definition of grid
step = 1.0 / resolution
x0 = np.asarray([-26.0+step, 34.5], dtype=np.float64)
size = (int(75.0/step), int(37.5/step))
    
# read sample data from file - stored as (lat, lon, val) tuple
obs_pts, obs_values = reader.read_csv_array('input/PressQFF_202007271200_' + str(num_points) + '.csv')

# the reference field is the accurate result from naive algorithm
print()
print('Computing naive Barnes interpolation which is baseline for RMSE')
naive_field = interpolation.barnes(obs_pts, obs_values.copy(), sigma, x0, step, size, method='naive')


# array to store RMSEs for convolution and optimized_convolution algorithm
res_RMSE = np.full((len(methods_set), len(num_iter_set)), 9999999.9, dtype=np.float64)

# RMSE measurements
for num_iter in num_iter_set:
    print()
    print('Number of iterations:', num_iter)
    print()
    
    
    # the loop over the different interpolation methods
    for method in methods_set:
        
        # copy first observation values since these might be slightly modified
        values = obs_values.copy()
        
        res_field = interpolation.barnes(obs_pts, values, sigma, x0, step, size,
            method=method, num_iter=num_iter, min_weight=0.0002)
        
        RMSE = rmse(naive_field, res_field, x0, step)
        
        print('%-24s  %6.4f' %(method, RMSE), flush=True)
        
        res_RMSE[methods_set.index(method), num_iter_set.index(num_iter)] = RMSE


# table output #####
print()
print()
print('RMSE in dependency of method and number of iterations')
print()

str = '    '
for j in range(len(methods_set)): str =  str + ' %10s' % method_short_names[j]
print(str)

for i in range(len(num_iter_set)):
    str = '%2d  ' % num_iter_set[i]
    for j in range(len(methods_set)): str = str + (' %10.4f' % res_RMSE[j,i])
    print(str)
    
    
# output figure 8 #####
x = num_iter_set
conv_RMSE = res_RMSE[0,:]
opt_conv_RMSE = res_RMSE[1,:]

fig, ax = plt.subplots(1, figsize=(8.0, 2.55), dpi=125)

ax.set_xlim(0.0, 51.0)
ax.set_ylim(0.0, 0.37)

ax.set_xticks(x)
plt.xticks(rotation=-90)

plt.grid()

ax.set_xlabel('Number of Convolutions')
ax.set_ylabel('RMSE')

# plot every second x-label
x_ticks = ax.xaxis.get_major_ticks()
for k in range(len(x_ticks)):
    if k % 2 == 0:
        x_ticks[k].label1.set_visible(False)

ax.set_yticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
# plot every second y-label
y_ticks = ax.yaxis.get_major_ticks()
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter("{:.1f}".format))
for k in range(len(y_ticks)):
    if k % 2 == 1:
        y_ticks[k].label1.set_visible(False)

ax.plot(x, conv_RMSE, c='#4260de', linewidth=1.1, label='original convolution')
plt.scatter(x, conv_RMSE, color='#4260de', s=22.2)

ax.plot(x, opt_conv_RMSE, c='#50c040',linewidth=1.1,  label='optimized convolution')
plt.scatter(x, opt_conv_RMSE, color='#50c040', s=22.2)

plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
