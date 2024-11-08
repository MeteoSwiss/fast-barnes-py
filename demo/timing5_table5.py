# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Runs the 'naive_S2' and the 'optimized_convolution_S2' Barnes interpolation algorithms
with a set of different resolutions and takes the respective execution
times. For the 'optimized_convolution' algorithm, the split times of the
convolution part and the resampling part are also recorded.
In order to receive more reliable results, the tests are executed several
times in a row and the best measured time is taken as final result.
The test results are printed as a summary table.

NOTE - THIS IS A ANNOYINGLY SLOW TEST
=====================================
The execution time of this program takes around 320 minutes.
You can reduce this time by decreasing the number of sample points to 872 or 218 for
instance.

Created on Sun May 29 2022, 19:45:01
@author: Bruno Zürcher
"""

import reader

from math import exp
import numpy as np

import time

from fastbarnes import interpolationS2


# the test parameters #########################################################

# the sample sizes, one of: [ 54, 218, 872, 3490 ]
num_points = 3490

# the different resolutions
resolutions_set = [ 4.0, 8.0, 16.0, 32.0, 64.0 ]

# the Gaussian width
sigma = 1.0

# the number of iterations
num_iter = 4


# other parameters, not necessarily to modify #################################

# the different interpolation methods
methods_set = [ 'naive_S2', 'optimized_convolution_S2' ]

# used for table outputs
method_short_names = [ 'naive_S2', 'opt_convol_S2' ]

# the number of measurement repetitions
num_rep = 5

# the rest time before measurement starts
sleep_time = 30


# do not modify after this point ##############################################

print('Barnes Interpolation Time Measurements 5')
print('****************************************')
print()
print('number of points: ', num_points)
print('sigma:            ', sigma)
print('iterations:       ', num_iter)


# compute weight that corresponds to specified max_dist
max_dist = 3.5
max_dist_weight = exp(-max_dist**2/2)

# give some time to put computer at maximum rest (e.g. install lock screen)
time.sleep(sleep_time)

# array to store best measured times
res_times = np.full((len(methods_set), len(resolutions_set)), 9999999.9, dtype=np.float64)

# array to store best split times for 'optimized_convolution_S2' method
split_times = np.full((2, len(resolutions_set)), 9999999.9, dtype=np.float64)

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
    size = (int(75.0/step), int(37.5/step))

    np_step = np.full(2, step, dtype=np.float64)
    np_sigma = np.full(2, sigma, dtype=np.float64)

    # test repetition loop
    for rep in range(num_rep):
        
        # the loop over the different interpolation methods
        for method in methods_set:
            
            # copy first observation values since these might be slightly modified
            values = obs_values.copy()
            
            # here we go: invoke Barnes interpolation and take execution time
            if method == 'naive_S2':
                # take usual code path
                start = time.perf_counter_ns()
                res_field = interpolationS2.barnes_S2(obs_pts, values, sigma, x0, step, size,
                    method=method, num_iter=num_iter)
                # execution time in seconds with accuracy of ms
                exec_time = ((time.perf_counter_ns() - start) // 1000000) / 1000.0
            
                print('%-25s %8.3f s' % (method, exec_time), flush=True)

                if exec_time < res_times[methods_set.index(method), resolutions_set.index(resolution)]:
                    res_times[methods_set.index(method), resolutions_set.index(resolution)] = exec_time
                
            else:           # i.e. for 'optimized_convolution_S2' method
                # do not directly invoke barnes_S2(), rather invoke its two sub-routines
                # separately and take split times
                start = time.perf_counter_ns()
                res1 = interpolationS2.interpolate_opt_convol_S2_part1(obs_pts, values,
                    np_sigma, x0, np_step, size, num_iter, max_dist_weight)
                # execution time in seconds with accuracy of ms
                split_time1 = ((time.perf_counter_ns() - start) // 1000000) / 1000.0
                
                start = time.perf_counter_ns()
                res_field = interpolationS2.interpolate_opt_convol_S2_part2(*res1)
                # execution time in seconds with accuracy of ms
                split_time2 = ((time.perf_counter_ns() - start) // 1000000) / 1000.0
                
                exec_time = split_time1 + split_time2
                
                print('%-25s %8.3f s (%5.3f s / %5.3f s)'
                    % (method, exec_time, split_time1, split_time2), flush=True)
            
                if exec_time < res_times[methods_set.index(method), resolutions_set.index(resolution)]:
                    res_times[methods_set.index(method), resolutions_set.index(resolution)] = exec_time
                    split_times[0, resolutions_set.index(resolution)] = split_time1
                    split_times[1, resolutions_set.index(resolution)] = split_time2


# -- output table 5 -----------------------------------------------------------
print()
print()
print('Best measured times')
print()

s = '%5s  %11s' % ('resol', 'grid size')
for i in range(len(methods_set)): s = s + (' %14s' % method_short_names[i])
s = s + '    %11s  %11s  %13s' % ('convol part', 'resamp part', 'lam grid size')
print(s)

for j in range(len(resolutions_set)):
    # compute grid sizes in dependency of resolution: 75°x37.5° and 64°x44°, resp.
    resolution = resolutions_set[j]
    s = '%5d  %4d x %4d' % (resolution, 75.0*resolution, 37.5*resolution)
    for i in range(len(methods_set)): s = s + (' %14.3f' % res_times[i,j])
    s = s + '    %11.3f  %11.3f    %4d x %4d' \
        % (split_times[0,j], split_times[1,j], 64.0*resolution, 44.0*resolution)
    print(s)
    