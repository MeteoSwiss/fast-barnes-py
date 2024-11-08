# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2024 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Unit test that checks the basic accumulation functionality used to compute a convolution
with a rect-kernel.

Created on Mon Oct 14 2024, 18:02:58
@author: Bruno Zürcher
"""

import unittest
import numpy as np
from math import sqrt, exp, pi
from fastbarnes import interpolation


class GaussianApproximationTest(unittest.TestCase):

    def assertEqualArrays(self, expected_arr, actual_arr, delta):
        """ Checks whether the specified arrays are identical up to the absolute difference delta. """
        self.assertEqual(len(expected_arr), len(actual_arr), 'arrays of different length')
        for k in range(len(expected_arr)):
            if abs(expected_arr[k] - actual_arr[k]) > delta:
                self.fail('index %d: expected %f and actual %f differ more than %f' %
                          (k, expected_arr[k], actual_arr[k], delta))

    def test_gaussian_1_dim(self):
        """ Tests whether 'convolution' method converges towards a Gaussian. """
        # test for various sigma values
        for sigma in [ 0.6, 0.8, 1.0, 1.5, 2.0 ]:
            # test for various number of iterations and corresponding maximum error differences
            for (num_iter, max_diff) in [ [3, 0.02395], [4, 0.01405], [5, 0.01232], [6, 0.01004] ]:
                resol = 512.0
                delta = 1.0/resol
                width = 3.5*sigma
                grid = np.arange(-width, width+0.000001, delta)

                # create Gaussian distribution for effective sigma
                sigma_eff = interpolation.get_sigma_effective(sigma, delta, num_iter)
                factor = 1.0 / sqrt(2.0 * pi) / sigma_eff
                gaussian = np.zeros(len(grid), dtype=np.float64)
                for k in range(len(grid)):
                    gaussian[k] = factor * exp(-(grid[k]/sigma_eff)**2/2)

                # create values and weight arrays with a 'finite' Dirac impulse in the center
                values = np.zeros(len(grid), dtype=np.float64)
                values[len(grid) // 2] = 1.0 / delta
                weights = np.copy(values)
                # create numpy equivalents as required by internal interpolation methods
                np_sigma = interpolation._to_np(sigma)
                np_delta = interpolation._to_np(delta)
                # compute convolution
                kernel_size = 2*interpolation._get_half_kernel_size(np_sigma, np_delta, num_iter) + 1
                interpolation._convolve_1d(values, weights, np_sigma, np_delta, (len(grid), ), kernel_size, num_iter, 9999.9)
                conv_scale_factor = (delta / 2 / sqrt(3/num_iter) / sigma_eff)**num_iter
                values *= conv_scale_factor

                # compare the two results
                self.assertEqualArrays(gaussian, values, max_diff / sigma_eff)

    def test_gaussian_1_dim_opt(self):
        """ Tests whether 'optimized_convolution' method converges towards a Gaussian. """
        # test for various sigma values
        for sigma in [ 0.6, 0.8, 1.0, 1.5, 2.0 ]:
            # test for various number of iterations and corresponding maximum error differences
            # error bounds are a tiny bit higher than in test_gaussian_1_dim() since tail value disturbs convergence
            # compared to that of a pure rect-kernel
            for (num_iter, max_diff) in [ [3, 0.02395], [4, 0.01405], [5, 0.012325], [6, 0.010045] ]:
                resol = 512.0
                delta = 1.0/resol
                width = 3.5*sigma
                grid = np.arange(-width, width+0.000001, delta)

                # create Gaussian distribution for sigma
                factor = 1.0 / sqrt(2.0 * pi) / sigma
                gaussian = np.zeros(len(grid), dtype=np.float64)
                for k in range(len(grid)):
                    gaussian[k] = factor * exp(-(grid[k]/sigma)**2/2)

                # create values and weight arrays with a 'finite' Dirac impulse in the center
                values = np.zeros(len(grid), dtype=np.float64)
                values[len(grid) // 2] = 1.0 / delta
                weights = np.copy(values)
                # create numpy equivalents as required by internal interpolation methods
                np_sigma = interpolation._to_np(sigma)
                np_delta = interpolation._to_np(delta)
                # compute convolution with a tailed rect-kernel
                kernel_size = 2*interpolation._get_half_kernel_size_opt(np_sigma, np_delta, num_iter) + 1
                tail_value = interpolation._get_tail_value(np_sigma, np_delta, num_iter)
                interpolation._convolve_tail_1d(values, weights, np_sigma, np_delta, (len(grid), ), kernel_size, num_iter, tail_value, 9999.9)
                conv_scale_factor = (delta / 2 / sqrt(3/num_iter) / sigma)**num_iter
                values *= conv_scale_factor

                # compare the two results
                self.assertEqualArrays(gaussian, values, max_diff / sigma)


if __name__ == '__main__':
    unittest.main()
