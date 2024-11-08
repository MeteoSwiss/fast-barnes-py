# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Module providing data reader functionality.

Created on Sat Jan 23 2022, 16:06:11
@author: Bruno Zürcher
"""

import struct
import numpy as np

###############################################################################

def read_csv_array(filename):
    """
    Reads the csv-file of observation data given line-wise by their latitude,
    longitude and observation value.
    The file header line defines the number of csv-lines and the number of values per line.

    Parameters
    ----------
    filename : string
        The file name.

    Returns
    -------
    tuple
        The lon-lat data array (numpy ndarray), the values (numpy ndarray).
    """
    f = open(filename, 'r')

    # read height and width
    (h, w) = map(int, f.readline().split(','))

    pts = np.empty((h, 2))
    values = np.empty(h)
    count = 0
    for line in f:
        res = line.split(',')
        pts[count,1] = float(res[0])
        pts[count,0] = float(res[1])
        values[count] = float(res[2])
        count += 1

    f.close()
    return (pts, values)


###############################################################################

def read_gridded_2darray(filename):
    """
    Reads an array of (Java-endian) doubles from a binary file, whereas integer height
    and width of array are initially read from file, as well as the grid specific
    parameters y0, x0, stepY, stepX.

    Parameters
    ----------
    filename : string
        The file name.

    Returns
    -------
    tuple
        The grid data array (numpy ndarray), y- and x-start-coordinate of grid, step in y- and x-direction.
    """
    f = open(filename, 'rb')

    # read dimensions
    block = f.read(8)
    (h, w) = struct.unpack('>2l', block)

    # read grid specifics
    block = f.read(32)
    (y0, x0, stepY, stepX) = struct.unpack('>4d', block)

    pack_struct = struct.Struct('>' + str(w) + 'd')
    num_bytes = 8*w
    container = []
    for j in range(h):
        block = f.read(num_bytes)
        data = pack_struct.unpack(block)
        container.append(data)

    f.close()
    return (np.array(container), y0, x0, stepY, stepX)
