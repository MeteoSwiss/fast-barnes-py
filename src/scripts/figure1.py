# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Plots the n-fold self-convolutions of a rectangle function as described in
section 2 of the paper and show how they nicely approach a Gaussian distribution.

Created on Sun May 29 20:28:02 2022
@author: Bruno Zürcher
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp, pi


###############################################################################

def gaussian(grid, sigma):
    """ Creates Gaussian distribution of specified sigma. """
    factor = 1.0 / sqrt(2.0 * pi) / sigma
    signal = np.zeros(len(grid))
    for k in range(len(grid)):
        signal[k] = factor * exp(-(grid[k]/sigma)**2/2)
    return signal
    
    
def uniform_distribution(n, sigma, delta):
    """ Creates uniform distribution u_n(x) as specified in paper. """
    bound = sqrt(3.0 / n) * sigma
    int_bound = int(bound / delta + 0.5)
    weight = 1.0 / (2*int_bound+1) / delta
    signal = np.zeros(2*int_bound+1)
    signal[:] = weight
    return signal


def convolve(signal_pdf, kernel_pdf, delta):
    """ Convolves the two specified centered signals defined on a grid with spacing delta. """
    signal_len = len(signal_pdf)
    kernel_len = len(kernel_pdf)
    result_len = signal_len+kernel_len-1
    result = np.zeros(result_len)
    
    # phase 1: kernel entering the signal
    for i in range(0, kernel_len-1):
        for k in range(i+1):
            result[i] += signal_pdf[i-k]*kernel_pdf[k]
    # phase 2: kernel within signal
    for i in range(kernel_len-1, result_len-kernel_len+1):
        for k in range(kernel_len):
            result[i] += signal_pdf[i-k]*kernel_pdf[k]
    # phase 3: kernel leaving the signal
    for i in range(result_len-kernel_len+1, result_len):
        for k in range(kernel_len-result_len+i, kernel_len):
            result[i] += signal_pdf[i-k]*kernel_pdf[k]
            
    # consider grid spacing, thus multiply with delta
    return result * delta
    
    
def center_signal(signal, target_len):
    """ Returns central part of signal of length target_len. """
    del_len = len(signal) - target_len
    if del_len < 0:
        s = np.zeros((-del_len)//2)
        return np.append(s, np.append(signal, s))
    else:
        return signal[del_len//2:len(signal)-del_len//2]


# the figure parameters #######################################################

sigma = 1.0

resol = 512.0
delta = 1.0/resol

width = 3.5

grid = np.arange(-width, width+0.000001, delta)


###############################################################################

gaussian = gaussian(grid, sigma)

n_list = [1, 2, 3, 4, 7, 10]

for n in n_list:
    
    # compute n-fold self-convolution of u_n(x)
    signal = uniform_distribution(n, sigma, delta)
    conv_signal = signal
    for k in range(1,n):
        conv_signal = convolve(conv_signal, signal, delta)
    conv_signal = center_signal(conv_signal, len(grid))
    
    # plot Gaussian and convolution figures
    plt.figure(figsize=(3.5, 5), dpi=150)
    plt.grid(b=True)
    plt.xlim(-3.5, 3.5)
    plt.ylim(0.0, 0.6)
    plt.xticks([-3,-2,-1,0,1,2,3])
    if n == 1:
        plt.xlabel('$u_{1}(x)$')
    else:
        label = r'$u_{%d}(x),  \,u_{%d}^{\,\ast%d}(x)$' % (n, n, n)
        plt.xlabel(label)

    plt.fill_between(grid, gaussian, 0, facecolor='#f4f4f4')
    plt.plot(grid, gaussian, c='#aaaaaa', linewidth=0.8)
    plt.plot(grid, center_signal(signal, len(grid)), c='#427cde', linewidth=1.2)
    if n > 1:
        plt.plot(grid, conv_signal, c='black', linewidth=1.0)
    
    plt.show(block=False)
    
    print('plotted self-convolution', n)
    
plt.show()
