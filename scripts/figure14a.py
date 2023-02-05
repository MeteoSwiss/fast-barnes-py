# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Plots a constellation of QFF values over Iceland that produces artifacts when
not treated with the quantization method.

Created on Sun Jun  5 19:35:48 2022
@author: Bruno Zürcher
"""

import reader
import plotmap

###############################################################################

# read observation data from file
obs_pts, obs_values = reader.read_csv_array('../input/obs/Iceland_PressOFF_constellation.csv')

# display station plot with values
plotmap.plot_Iceland_station_map(obs_pts, obs_values)
