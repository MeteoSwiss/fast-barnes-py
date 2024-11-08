# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Runs the 'convolution' and 'optimized_convolution' Barnes interpolation algorithms
with a set of different number of iterations and takes the respective execution
times. In order to receive more reliable results, the tests are executed several
times in a row and the best measured time is taken as final result.
The test results are printed with further interesting information as a summary
table and plotted in form of a graph.

NOTE
====
The execution time of this program takes around 6 minutes.
You can reduce this time by decreasing the resolution to 16.0 or 8.0 for instance.

Created on Sun May 29 2022, 20:03:40
@author: Bruno Zürcher
"""

import reader
from rmse import rmse

import numpy as np
import matplotlib.pyplot as plt

import time

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
num_iter_set = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50 ]

# the number of measurement repetitions
num_rep = 5

# the rest time before measurement starts
sleep_time = 30


# do not modify after this point ##############################################

print('Barnes Interpolation Time Measurements 4')
print('****************************************')
print()
print('number of points: ', num_points)
print('resolution:       ', resolution)
print('sigma:            ', sigma)


# definition of grid
step = 1.0 / resolution
x0 = np.asarray([-26.0+step, 34.5], dtype=np.float64)
size = (int(37.5/step), int(75.0/step))
    
# read sample data from file
obs_pts, obs_values = reader.read_csv_array('input/PressQFF_202007271200_' + str(num_points) + '.csv')

# the reference field is the accurate result from naive algorithm A
print()
print("Computing naive Barnes interpolation which is baseline for RMSE")
naive_field = interpolation.barnes(obs_pts, obs_values.copy(), sigma, x0, step, size, method='naive')


# no need for sleep time, since naive field computation took already long...
# give some time to put computer at maximum rest (e.g. install lock screen)
# time.sleep(sleep_time)

# array to store best measured times
res_times = np.full((len(methods_set), len(num_iter_set)), 9999999.9, dtype=np.float64)

# array to store RMSEs for convolution and optimized_convolution algorithm B
res_RMSE = np.full((len(methods_set), len(num_iter_set)), 9999999.9, dtype=np.float64)

# take time measurements
for num_iter in num_iter_set:
    print()
    print('Number of iterations:', num_iter)
    print()
        
    
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
            
            print('%-24s %8.3f s' % (method, exec_time), flush=True)
            
            # save best execution times
            if exec_time < res_times[methods_set.index(method), num_iter_set.index(num_iter)]:
                res_times[methods_set.index(method), num_iter_set.index(num_iter)] = exec_time
                
            # compute and save RMSE
            if rep == 0:
                RMSE = rmse(naive_field, res_field, x0, step)
                res_RMSE[methods_set.index(method), num_iter_set.index(num_iter)] = RMSE


# -- output table 4 -----------------------------------------------------------
print()
print()
print('Result summary')
print()


s = '       ' + ('%32s       %32s' % (methods_set[0], methods_set[1]))
print(s)
s = '       ' \
    + ('%2s %9s %9s %9s       ' % ('T', 'sigma_eff', 'RMSE', 't_exec'))
s = s + ('%2s %9s %9s %9s' % ('T', 'tail_val', 'RMSE', 't_exec'))
print(s)

for num_iter in num_iter_set:
    ind = num_iter_set.index(num_iter)
    s = '%2d     ' % num_iter
    s = s + ('%2d %9.4f %9.4f %9.3f       ' % (
        interpolation.get_half_kernel_size(sigma, step, num_iter),
        interpolation.get_sigma_effective(sigma, step, num_iter),
        res_RMSE[0,ind],
        res_times[0,ind]))
    s = s + ('%2d %9.4f %9.4f %9.3f' % (
        interpolation.get_half_kernel_size_opt(sigma, step, num_iter),
        interpolation.get_tail_value(sigma, step, num_iter),
        res_RMSE[1,ind],
        res_times[1,ind]))
    print(s)


# -- output figure 7 ----------------------------------------------------------
# but only range 1-10 and assuming presence of full range 1-10
x = num_iter_set[:10]
conv_times = res_times[0,:10]
opt_conv_times = res_times[1,:10]

fig, ax = plt.subplots(1, figsize=(7.2, 3.4), dpi=150)

ax.set_xlim(0.75, 10.25)
ax.set_ylim(0.145, 0.405)

ax.set_xticks(x)

plt.grid()

ax.set_xlabel('Number of Convolutions')
ax.set_ylabel('Execution Time [s]')

ax.plot(x, conv_times, c='#4260de', label='original convolution')
plt.scatter(x, conv_times, color='#4260de')

ax.plot(x, opt_conv_times, c='#50c040', label='optimized convolution')
plt.scatter(x, opt_conv_times, color='#50c040')

plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
