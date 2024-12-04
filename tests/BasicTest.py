# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2024 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Unit test that checks the basic accumulation functionality used to compute a convolution
with a rect-kernel.

Created on Sat Oct 12 2024, 15:15:25
@author: Bruno Zürcher
"""

import unittest
import numpy as np
from fastbarnes import interpolation


class BasicTest(unittest.TestCase):

    # def assertArrayValueRange2(self, arr, min_val, max_val, eps):
    #     """ Checks whether all entries of arr are contained within [min_val, max_val] or equal to np.nan. """
    #     for k in range(len(arr)):
    #         if math.isnan(arr[k]):
    #             continue
    #         if arr[k] >= min_val-eps and arr[k] <= max_val+eps:
    #             continue
    #         self.fail('unexpected array element: ' + str(arr[k]))


    def assertArrayValueRange(self, arr, min_val, max_val, eps):
        """ Checks whether all entries of arr are contained within [min_val, max_val] or equal to np.nan. """
        self.assertTrue(
            np.all(
                   np.logical_or(np.logical_and(arr >= min_val-eps, arr <= max_val+eps),
                   np.isnan(arr))),
            'elements should be contained in range ['+str(min_val)+','+str(max_val)+'] or equal to np.nan')


    def test_1d_interpolation(self):
        """ Just invokes an interpolation to see it terminating successfully and with a reasonable value range.  """
        step = 1.0 / 64
        size = 129

        pts = np.asarray([ 0.15, 0.2, 0.33, 0.35, 0.46, 0.59, 0.61, 0.66, 0.83, 0.98, 1.21, 1.29, 1.4, 1.57, 1.6, 1.79])
        val = np.asarray([ 1.0,  0.0, 1.0,  1.0,  0.0,  1.0,  1.0,  0.0,  0.0,  1.0,  0.0,  1.0,  1.0, 1.0,  0.0, 1.0])

        for method in [ 'convolution', 'optimized_convolution' ]:
            for sigma in [ 0.5, 0.2, 0.05 ]:
                for num_iter in [ 1, 2, 3, 4, 5, 6, 8, 10 ]:
                    result = interpolation.barnes(pts, val, sigma, 0.0, step, size, method=method, num_iter=num_iter)
                    self.assertArrayValueRange(result, 0.0, 1.0, 1e-15)


    def test_2d_interpolation(self):
        """ Just invokes an interpolation to see it terminating successfully and with a reasonable value range.  """
        step = 1.0 / 64
        size = (129, 129)

        # encode input data as: x-coor, y-coor, value
        data = np.asarray([
            [0.15,0.07,1.0], [0.22,0.11,0.0], [0.49,0.21,1.0], [0.63,0.09,0.0], [0.94,0.20,1.0],
            [1.02,0.18,1.0], [1.11,0.04,0.0], [1.38,0.18,0.0], [1.74,0.24,1.0], [1.87,0.21,1.0],
            [0.10,0.47,0.0], [0.31,0.51,0.0], [0.40,0.31,1.0], [0.70,0.49,0.0], [0.91,0.50,1.0],
            [1.16,0.38,1.0], [1.19,0.34,0.0], [1.35,0.48,0.0], [1.69,0.46,1.0], [1.80,0.36,0.0],
            [0.04,0.67,1.0], [0.37,0.81,1.0], [0.59,0.71,1.0], [0.88,0.59,0.0], [0.93,0.81,1.0],
            [1.01,0.88,1.0], [1.26,0.84,0.0], [1.44,0.78,0.0], [1.52,0.86,1.0], [1.91,0.66,0.0],
            [0.14,0.99,0.0], [0.17,1.05,1.0], [0.38,1.21,1.0], [0.43,0.96,0.0], [0.83,1.11,1.0],
            [1.00,1.00,1.0], [1.13,0.98,0.0], [1.24,1.23,1.0], [1.48,1.16,1.0], [1.60,1.08,0.0],
            [0.34,1.28,1.0], [0.37,1.47,1.0], [0.68,1.37,0.0], [0.93,1.52,0.0], [0.97,1.39,1.0],
            [1.03,1.50,1.0], [1.44,1.42,0.0], [1.54,1.51,1.0], [1.78,1.49,0.0], [1.89,1.60,1.0],
            [0.17,1.79,1.0], [0.31,1.95,0.0], [0.55,1.63,0.0], [0.85,1.78,1.0], [0.92,1.65,1.0],
            [1.07,1.90,0.0], [1.30,1.82,1.0], [1.46,1.71,1.0], [1.73,1.69,0.0], [1.91,1.91,1.0],
        ])
        pts = data[:, 0:2]
        val = data[:, 2]

        for method in [ 'convolution', 'optimized_convolution' ]:
            for sigma in [ 0.5, 0.2, 0.05 ]:
                for num_iter in [ 1, 2, 3, 4, 5, 6, 8, 10 ]:
                    result = interpolation.barnes(pts, val, sigma, 0.0, step, size, method=method, num_iter=num_iter)
                    self.assertArrayValueRange(result, 0.0, 1.0, 5e-13)



    def test_3d_interpolation(self):
        """ Just invokes an interpolation to see it terminating successfully and with a reasonable value range.  """
        step = 1.0 / 64
        size = (129, 129, 129)

        num_values = 307
        pts = 2.0 * np.random.rand(num_values, 3)
        val = np.random.randint(2, size=num_values).astype(dtype=np.float64)

        for method in [ 'convolution', 'optimized_convolution' ]:
            for sigma in [ 0.5, 0.2, 0.05 ]:
                for num_iter in [ 1, 3, 4, 6, 10 ]:
                    result = interpolation.barnes(pts, val, sigma, 0.0, step, size, method=method, num_iter=num_iter)
                    self.assertArrayValueRange(result, 0.0, 1.0, 1e-12)



    def test_array_width_vs_kernel_size(self):
        """ Tests whether a too small data array size leads to a failure. """
        sigma = 0.5
        step = 1.0 / 32

        # tests for method 'convolution'
        for num_iter in [ 3, 4, 6 ]:
            kernel_size = 2*interpolation.get_half_kernel_size(sigma, step, num_iter) + 1

            pts = np.asarray([ 0.2, 0.4, 0.7])
            val = np.asarray([ 18.5, 17.0, 19.25])

            # use array size that is big enough
            result = interpolation.barnes(pts, val, sigma, 0.0, step, kernel_size + 1,
                method='convolution', num_iter=num_iter)

            # use array size that is too small
            try:
                result = interpolation.barnes(pts, val, sigma, 0.0, step, kernel_size,
                    method='convolution', num_iter=num_iter)
                self.fail('test case should be raised as a RuntimeError')
            except RuntimeError:
                # do nothing
                continue

        # tests for method 'optimized_convolution'
        for num_iter in [ 3, 4, 6 ]:
            kernel_size = 2*interpolation.get_half_kernel_size_opt(sigma, step, num_iter) + 1

            pts = np.asarray([ 0.2, 0.4, 0.7])
            val = np.asarray([ 18.5, 17.0, 19.25])

            # use array size that is big enough
            result = interpolation.barnes(pts, val, sigma, 0.0, step, kernel_size + 1,
                method='optimized_convolution', num_iter=num_iter)

            # use array size that is too small
            try:
                result = interpolation.barnes(pts, val, sigma, 0.0, step, kernel_size,
                    method='optimized_convolution', num_iter=num_iter)
                self.fail('test case should be raised as a RuntimeError')
            except RuntimeError:
                # do nothing
                continue


if __name__ == '__main__':
    unittest.main()
