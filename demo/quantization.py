# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Module that implements a well performing bit quantization algorithm by using
the Numba just-in-time compiler.

Created on Sat Jun  4 21:59:37 2022
@author: Bruno Zürcher
"""

import numpy as np
from math import frexp, ldexp
from numba import njit

###############################################################################

@njit
def quant(arr, nbits):
    """ Removes the nbits least significant bits of the mantissa of the specified float64 numbers. """
    if nbits <= 0:  return
    shiftbits = 53-nbits
    # create 1d view of arr
    barr = np.ravel(arr)
    for i in range(len(barr)):
        # quick test for not NaN:
        if barr[i] == barr[i]:
            m, e = frexp(barr[i])
            m = int(m*2**shiftbits + 0.5)
            barr[i] = ldexp(m, e-shiftbits)


###############################################################################

# some code to test the functionality

if __name__ == '__main__':
    
    def to_bin_m_e(x):
        """ Return String with binary representation of mantissa and exponent. """
        m, e = frexp(x)
        # neg = m < 0.0
        # if neg: m = -m
        m = int(m * 2**53)
        s = ''
        while m != 0 and m != -1:
            s = ('1' if m & 1 else '0') + s
            m //= 2
        return s + ' ' + str(e)
    
    from math import sqrt


    # quantization example with some irrational numbers
    arr = np.empty(3, dtype=np.float64)
    for k in range(14):
        arr[0] = np.pi
        arr[1] = np.e
        arr[2] = sqrt(2.0)
        
        quant(arr, k)
        
        for i in range(len(arr)):
            print('%2d   %16.14f   %s' % (k, arr[i], to_bin_m_e(arr[i])))
        print()
    
    
    # quantization example with 'almost 2' numbers
    arr = np.empty(6, dtype=np.float64)
    for k in range(5):
        for i in range(len(arr)):
            q = 1.0
            arr[i] = 0.0
            for j in range(50+i):
                arr[i] += q
                q = q / 2.0
            # arr[i] = ((1 << (50+i)) - 1) / (2.0 ** (50+i))
        
        if k >= 0:
            quant(arr, k)
        
        for i in range(len(arr)):
            print('%2d   %16.14f   %s' % (k, arr[i], to_bin_m_e(arr[i])))
        print()
