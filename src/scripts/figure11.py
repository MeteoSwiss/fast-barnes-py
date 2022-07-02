# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Plots the map scale in dependency of the latitude for a Lambert conformal
conical projection with standard parallels at 42.5°N and 65.5°N.

Created on Sat Jun  4 17:29:50 2022
@author: Bruno Zürcher
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

import interpolationS2
import lambert_conformal

# compute Lambert scale factors ###############################################

x = np.asarray([ 34.5+k*0.5 for k in range(76) ])

lambert_proj = interpolationS2.get_lambert_proj()

scales = lambert_conformal.get_scale(x, *lambert_proj)

# do not modify after this point ##############################################

fig, ax = plt.subplots(1, figsize=(7.2, 3.4), dpi=150)

ax.set_xlim(32.0, 74.0)
ax.set_ylim(0.97, 1.05)

ax.set_xticks([35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0])
plt.grid()

# plt.title('title')
ax.set_xlabel('Latitude [°N]')
ax.set_ylabel('Effective Map Scale')

plt.plot(x, scales, c='#4260de', label='real scale')
plt.plot([30.0,75.0], [1.0,1.0], c='#404040', linewidth=0.75, label="unit scale")

# # filled areas
# plt.fill_between(x[:17], scales[:17], 1.0, facecolor='#e0ffe0')
# plt.fill_between(x[16:63], scales[16:63], 1.0, facecolor='#ffe0e0')
# plt.fill_between(x[62:], scales[62:], 1.0, facecolor='#e0ffe0')

# plt.scatter(x, scales, color='blue')

# plt.legend(loc='lower right')

plt.tight_layout()
plt.show()
