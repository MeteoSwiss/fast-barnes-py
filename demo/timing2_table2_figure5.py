# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Runs the 'naive', the 'radius' and the 'convolution' Barnes interpolation algorithms
with a set of different number of resolutions and takes the respective execution
times. In order to receive more reliable results, the tests are executed several
times in a row and the best measured time is taken as final result.
The test results are printed as a summary table and plotted in form of a graph.

NOTE - THIS IS A ANNOYINGLY SLOW TEST
=====================================
The execution time of this program takes around 140 minutes.
You can reduce this time by decreasing the number of sample points to 872 or 218
for instance.

Created on Sat Jun 11 16:32:15 2022
@author: Bruno Zürcher
"""

import reader

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from math import sqrt

import time

from fastbarnes import interpolation


# the test parameters #########################################################

# one of: [ 54, 218, 872, 3490 ]
num_points = 3490

# the different resolutions
resolutions_set = [ 4.0, 8.0, 16.0, 32.0, 64.0 ]

# one of: [ 0.25, 0.5, 1.0, 2.0, 4.0 ]
sigma = 1.0

# one of: [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50 ]
num_iter = 4


# other parameters, not necessarily to modify #################################

# the different interpolation methods
methods_set = [ 'naive', 'radius', 'convolution' ]

# used for table outputs
method_short_names = [ 'naive', 'radius', 'convol.' ]

# the number of measurement repetitions
num_rep = 5

# the rest time before measurement starts
sleep_time = 30


# do not modify after this point ##############################################

def x_formatter(x, pos):
    """ Formatter that draws the x-axis tick mark labels. """
    if pos == None:
        return ('%d' % int(x))
    if pos < len(resolutions_set):
        fac = int(sqrt(x/2))
        return str(2*fac)+' x '+str(fac)
    if x == 100000:
        return '0.1 Mio'
    if x == 1000000:
        return '1 Mio'
    if x == 10000000:
        return '10 Mio'


def y_formatter(y, pos):
    """ Formatter that draws the y-axis tick mark labels. """
    if pos == None:
        return ('%.2f' % y)
    if y >= 1:
        return str(int(y))
    if y == 0.1:
        return '0.1'
    if y == 0.01:
        return '0.01'
    if y == 0.001:
        return '0.001'


###############################################################################

print('Barnes Interpolation Time Measurements 2')
print('****************************************')
print()
print('sample points: ', num_points)
print('sigma:         ', sigma)
print('iterations:    ', num_iter)


# give some time to put computer at maximum rest (e.g. install lock screen)
time.sleep(sleep_time)

# array to store best measured times
res_times = np.full((len(methods_set), len(resolutions_set)), 9999999.9, dtype=np.float64)

# take time measurements
for resolution in resolutions_set:
    print()
    print('Resolution:', resolution)
    print()
    
    # read sample data from file
    obs_pts, obs_values = reader.read_csv_array('input/PressQFF_202007271200_' + str(num_points) + '.csv')
    
    # definition of grid
    step = 1.0 / resolution
    x0 = np.asarray([-26.0+step, 34.5], dtype=np.float64)
    size = (int(37.5/step), int(75.0/step))
    
    
    # test repetition loop
    for rep in range(num_rep):
        
        # the loop over the different interpolation methods
        for method in methods_set:
            
            # copy first observation values since these might be slightly modified
            values = obs_values.copy()
            
            # here we go: invoke Barnes interpolation and take execution time
            start = time.perf_counter_ns()
            res_field = interpolation.barnes(obs_pts, values, sigma, x0, step, size,
                method=method, num_iter=num_iter)
            # execution time in seconds with accuracy of ms
            exec_time = ((time.perf_counter_ns() - start) // 1000000) / 1000.0
            
            print('%-12s %8.3f s' % (method, exec_time), flush=True)
            
            if exec_time < res_times[methods_set.index(method), resolutions_set.index(resolution)]:
                res_times[methods_set.index(method), resolutions_set.index(resolution)] = exec_time


# -- output table 2 -----------------------------------------------------------
print()
print()
print('Best measured times')
print()


width_set = [int(75.0*res) for res in resolutions_set]
height_set = [int(37.5*res) for res in resolutions_set]

s = '%4s   %4s   %4s' % (' ', ' ', ' ')
for i in range(len(methods_set)): s = s + (' %8s' % method_short_names[i])
print(s)

for j in range(len(resolutions_set)):
    s = '%4d x %4d   %4.1f' % (width_set[j], height_set[j], resolutions_set[j])
    for i in range(len(methods_set)): s = s + (' %8.3f' % res_times[i,j])
    print(s)
    
    
# -- output figure 5 ----------------------------------------------------------
fig, ax = plt.subplots(1, figsize=(7.2, 3.4), dpi=150)

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlim(40000.0, 14000000.0)
ax.set_ylim(0.0005, 2000.0)

x_pos = [ width_set[i]*height_set[i] for i in range(len(width_set)) ]

ax.set_xticks(x_pos + [ 100000.0, 1000000.0, 10000000.0 ])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(x_formatter))
plt.xticks(rotation=-90)

ax.set_yticks([ 0.001, 0.01, 0.1, 1, 10, 100, 1000 ])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(y_formatter))
plt.grid()

# supress grid lines at x-data value positions, which are the first few ones in list
lineList = ax.get_xgridlines()
for k in range(len(x_pos)):
    lineList[k].set_color('w')
    lineList[k].set_linewidth(0.0)

ax.set_xlabel('Number of Grid Points')
ax.set_ylabel('Execution Time [s]')

# define colors
color_set = ['#db060b', '#c878bc', '#4260de' ]

for method in methods_set:
    method_index = methods_set.index(method)
    ax.plot(x_pos, res_times[method_index], c=color_set[method_index], label=method)

for method in methods_set:
    method_index = methods_set.index(method)
    plt.scatter(x_pos, res_times[method_index], c=color_set[method_index])

plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
