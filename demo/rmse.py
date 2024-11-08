# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Module that provides functions to compute the root-mean-square error (RMSE).

Created on Sat Jun 4 2022, 09:56:19
@author: Bruno Zürcher
"""

import numpy as np


###############################################################################

def rmse(naive_field, other_field, x0, step):
    """ 
    Returns the RMSE of the Western Europe subdomain when comparing the
    specified field with the exact naive field.
    """
    # the corner indices for the Western Europe subdomain for which we compute RMSE
    # i.e. for [-7.0°, 5.0°] x [36.0°, 56.0°]
    ll = [ int((-7.0-x0[0]+step)/step), int((36.0-x0[1])/step)]
    ur = [ int((5.0-x0[0]+step)/step), int((56.0-x0[1])/step)]

    RMSE = np.sqrt(np.nanmean((
        other_field[ll[1]:ur[1],ll[0]:ur[0]] - naive_field[ll[1]:ur[1],ll[0]:ur[0]])**2))
    
    return RMSE
    