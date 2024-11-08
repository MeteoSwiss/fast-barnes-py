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


class AccumulationTest(unittest.TestCase):

    def assertArrayEqualValue(self,expected_val,actual_arr):
        """ Checks whether all entries of actual_arr are equal to specified value. """
        for k in range(len(actual_arr)):
            self.assertEqual(expected_val, actual_arr[k])

    def assertEqualArrays(self, expected_arr, actual_arr):
        """ Checks whether the specified arrays are identical. """
        self.assertEqual(len(expected_arr), len(actual_arr), 'arrays of different length')
        for k in range(len(expected_arr)):
            self.assertEqual(expected_arr[k], actual_arr[k])

    def test_accumulate_array_1_fold(self):
        """ Tests whether a simple 1-fold convolution with _accumulate_array() is correct. """
        size = 32
        h_arr = np.empty(size, dtype=np.float64)

        # value completely inside array
        in_arr = np.zeros(size, dtype=np.float64)
        in_arr[10] = 1
        out_arr = interpolation._accumulate_array(in_arr, h_arr, size, 9, 1)
        self.assertArrayEqualValue(0, out_arr[:6])
        self.assertArrayEqualValue(1, out_arr[6:15])
        self.assertArrayEqualValue(0, out_arr[15:])

        # value at left border
        in_arr = np.zeros(size, dtype=np.float64)
        in_arr[2] = 1
        out_arr = interpolation._accumulate_array(in_arr, h_arr, size, 9, 1)
        self.assertArrayEqualValue(1, out_arr[:7])
        self.assertArrayEqualValue(0, out_arr[7:])

        # value at right border
        in_arr = np.zeros(size, dtype=np.float64)
        in_arr[30] = 1
        out_arr = interpolation._accumulate_array(in_arr, h_arr, size, 9, 1)
        self.assertArrayEqualValue(0, out_arr[:26])
        self.assertArrayEqualValue(1, out_arr[26:])

    def test_accumulate_array_2_fold(self):
        """ Tests whether a simple 2-fold convolution with _accumulate_array() is correct. """
        size = 32
        h_arr = np.empty(size, dtype=np.float64)

        # value completely inside array
        in_arr = np.zeros(size, dtype=np.float64)
        in_arr[16] = 1
        out_arr = interpolation._accumulate_array(in_arr, h_arr, size, 9, 2)
        self.assertArrayEqualValue(0, out_arr[:8])
        self.assertEqualArrays(np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1]), out_arr[8:25])
        self.assertArrayEqualValue(0, out_arr[25:])

        # value at left border
        in_arr = np.zeros(size, dtype=np.float64)
        in_arr[2] = 1
        out_arr = interpolation._accumulate_array(in_arr, h_arr, size, 9, 2)
        self.assertEqualArrays(np.asarray([5, 6, 7, 7, 7, 6, 5, 4, 3, 2, 1]), out_arr[:11])
        self.assertArrayEqualValue(0, out_arr[11:])

        # value at right border
        in_arr = np.zeros(size, dtype=np.float64)
        in_arr[30] = 1
        out_arr = interpolation._accumulate_array(in_arr, h_arr, size, 9, 2)
        self.assertArrayEqualValue(0, out_arr[:22])
        self.assertEqualArrays(np.asarray([1, 2, 3, 4, 5, 6, 6, 6, 6, 5]), out_arr[22:])

    def test_accumulate_tail_array_1_fold(self):
        """ Tests whether a simple 1-fold convolution with _accumulate_tail_array() is correct. """
        size = 32
        h_arr = np.empty(size, dtype=np.float64)

        # value completely inside array
        in_arr = np.zeros(size, dtype=np.float64)
        in_arr[10] = 1
        out_arr = interpolation._accumulate_tail_array(in_arr, h_arr, size, 7, 1, 0.25)
        self.assertArrayEqualValue(0, out_arr[:6])
        self.assertEqualArrays(np.asarray([0.25, 1, 1, 1, 1, 1, 1, 1, 0.25]), out_arr[6:15])
        self.assertArrayEqualValue(0, out_arr[15:])

        # value at left border
        in_arr = np.zeros(size, dtype=np.float64)
        in_arr[2] = 1
        out_arr = interpolation._accumulate_tail_array(in_arr, h_arr, size, 7, 1, 0.25)
        self.assertEqualArrays(np.asarray([1, 1, 1, 1, 1, 1, 0.25]), out_arr[:7])
        self.assertArrayEqualValue(0, out_arr[7:])

        # value at right border
        in_arr = np.zeros(size, dtype=np.float64)
        in_arr[30] = 1
        out_arr = interpolation._accumulate_tail_array(in_arr, h_arr, size, 7, 1, 0.25)
        self.assertArrayEqualValue(0, out_arr[:26])
        self.assertEqualArrays(np.asarray([0.25, 1, 1, 1, 1, 1]), out_arr[26:])

    def test_accumulate_tail_array_2_fold(self):
        """ Tests whether a simple 2-fold convolution with _accumulate_tail_array() is correct. """
        size = 32
        h_arr = np.empty(size, dtype=np.float64)

        # value completely inside array
        in_arr = np.zeros(size, dtype=np.float64)
        in_arr[16] = 1
        out_arr = interpolation._accumulate_tail_array(in_arr, h_arr, size, 7, 2, 0.5)
        self.assertArrayEqualValue(0, out_arr[:8])
        self.assertEqualArrays(np.asarray([0.25, 1, 2, 3, 4, 5, 6, 7, 7.5, 7, 6, 5, 4, 3, 2, 1, 0.25]), out_arr[8:25])
        self.assertArrayEqualValue(0, out_arr[25:])

        # value at left border
        in_arr = np.zeros(size, dtype=np.float64)
        in_arr[2] = 1
        out_arr = interpolation._accumulate_tail_array(in_arr, h_arr, size, 7, 2, 0.5)
        self.assertEqualArrays(np.asarray([4.5, 5.5, 6.25, 6.5, 6, 5, 4, 3, 2, 1, 0.25]), out_arr[:11])
        self.assertArrayEqualValue(0, out_arr[11:])

        # value at right border
        in_arr = np.zeros(size, dtype=np.float64)
        in_arr[30] = 1
        out_arr = interpolation._accumulate_tail_array(in_arr, h_arr, size, 7, 2, 0.5)
        self.assertArrayEqualValue(0, out_arr[:22])
        self.assertEqualArrays(np.asarray([0.25, 1, 2, 3, 4, 5, 5.5, 5.5, 5.25, 4.5]), out_arr[22:])

    def test_accumulate_array_versions(self):
        """
        Tests whether _accumulate_array() and a _accumulate_tail_array() call with trivial tail
        result in identical arrays.
        """
        size = 32
        h_arr = np.empty(size, dtype=np.float64)

        in_arr = np.zeros(size, dtype=np.float64)
        in_arr[3] = 1
        in_arr[9] = 2.5
        in_arr[21] = -1.25
        in_arr[30] = 1

        # compute 3-fold convolutions
        # simple case with tail value 0, with equal rect-kernel size
        out_arr1 = interpolation._accumulate_array(np.copy(in_arr), h_arr, size, 9, 3)
        out_arr2 = interpolation._accumulate_tail_array(np.copy(in_arr), h_arr, size, 9, 3, 0)
        self.assertEqualArrays(out_arr1, out_arr2)

        # simple case with tail value 1, but shorter rect-kernel size
        out_arr1 = interpolation._accumulate_array(np.copy(in_arr), h_arr, size, 9, 3)
        out_arr2 = interpolation._accumulate_tail_array(np.copy(in_arr), h_arr, size, 7, 3, 1)
        self.assertEqualArrays(out_arr1, out_arr2)


if __name__ == '__main__':
    unittest.main()
