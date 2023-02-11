# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Module that contains different auxiliary plot functions that were used to
visualize the results presented in the paper.

Created on Sat Jan 23 14:51:40 2021
@author: Bruno Zürcher
"""

###############################################################################
# required code to make Basemap library work in Anaconda - refer to stacktrace

# import os
# os.environ["PROJ_LIB"] = os.path.join(os.environ["CONDA_PREFIX"], "share", "proj")

###############################################################################

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg

from PIL import Image

from mpl_toolkits.basemap import Basemap

import numpy as np

from fastbarnes.util import lambert_conformal

###############################################################################

def plot_lat_lon_map(data, x0, step, size, scatter_pts=None, alpha_channel=True,
    line_labels=True, more_parallels=False):
    """
    Plots an isoline visualization of the specified data field on top of a
    Europe geography background using a lon-lat coordinate system.
    """
    # create mesh grid
    gridX = np.arange(0.0, size[1], 1.0)
    gridX = gridX*step + x0[0]
    # y-axis of grid
    gridY = np.arange(0.0, size[0], 1.0)
    gridY = gridY*step + x0[1]
    # arrays holding x- and y-coordinates of grid points
    X, Y = np.meshgrid(gridX, gridY)


    fig = plt.figure(figsize=(16, 12), edgecolor='w', dpi=72)
    # resolution applies only for vector graphic; 'c' and 'l' too low
    m = Basemap(projection='cyl', resolution='i',
        llcrnrlat=34.5, urcrnrlat=72, llcrnrlon=-26, urcrnrlon=49)
    
    # default background
    m.shadedrelief(scale=1.0)

    # other background maps can be downloaded from
    # http://www.naturalearthdata.com/downloads/10m-raster-data/10m-cross-blend-hypso/
    # m.warpimage(image='C:...PATH.../Maps/HYP_HR/HYP_HR_SR_W_DR.tif')

    m.drawcountries(linewidth=0.4)
    if more_parallels:
        m.drawparallels([20.0,30.0,40.0,42.5,50.0,60.0,65.5,70.0,80.0],labels=[1,0,0,0])
    else:
        m.drawparallels([20.0,30.0,40.0,50.0,60.0,70.0,80.0],labels=[1,0,0,0])
    m.drawmeridians(np.arange(0.,361.,10.),labels=[0,0,0,1])

    # the levels of the plotted isolines
    levels = np.arange(976, 1026, 2)
    cs = m.contour(X, Y, data, levels, latlon=True, linewidths=1.0, colors='black')
    if line_labels:
        # plot the line labels
        plt.clabel(cs, levels[::2], inline=True, fmt='%d', fontsize=11, colors='black')


    if alpha_channel:
        # define mask
        mask = np.isnan(data)
        alpha_channel = mask*0.35
        cmap = ListedColormap(['black', 'black'])
        m.imshow(mask, alpha=alpha_channel, cmap=cmap)


    if not scatter_pts is None:
        xcoor = scatter_pts[:,0]
        ycoor = scatter_pts[:,1]
        m.scatter(xcoor, ycoor, color='red', s=9, marker='.')
    
    plt.show()


###############################################################################

def plot_Lambert_map(data, lam_x0, step, lam_size, lam_proj, scatter_pts=None,
    alpha_channel=True):
    """
    Plots an isoline visualization of the specified data field on top of a
    Europe geography background using a Lambert conformal coordinate system.
    """
    # create mesh grid
    gridX = np.arange(0.0, lam_size[1], 1.0)
    gridX = gridX*step + lam_x0[0]
    # y-axis of grid
    gridY = np.arange(0.0, lam_size[0], 1.0)
    gridY = gridY*step + lam_x0[1]
    # arrays holding x- and y-coordinates of grid points in Lambert space
    X, Y = np.meshgrid(gridX, gridY)

    # map mesh grid to lonlat system
    lambert_conformal.to_geo2(X, Y, X, Y, *lam_proj)


    fig = plt.figure(figsize=(16, 8.25), edgecolor='w', dpi=72)
    # resolution applies only for vector graphic; 'c' and 'l' too low
    m = Basemap(projection='lcc', lat_1=42.5, lat_2=65.5, lat_0=34.5, lon_0=11.5, rsphere=180.0/np.pi, resolution='i',
        llcrnrlat=28.14, llcrnrlon=-21.0, urcrnrlat=58.5, urcrnrlon=79.5)
    # CAVEAT: the two Lambert projections m and lam_proj are formally identical
    # but to move from one to the other an affine map has to be applied
    
    
    # compute the subrange of the required alpha image:
    # map ll and ur corner of m to Lambert space defined by lam_proj
    p = np.asarray([[-21.0, 28.14]])
    ll = lambert_conformal.to_map(p, p.copy(), *lam_proj)[0]
    p = np.asarray([[79.5, 58.5]])
    ur = lambert_conformal.to_map(p, p.copy(), *lam_proj)[0]
    
    # get the corresponding ll and ur indices in data array
    iOff = int((ll[0]-lam_x0[0])/step + 0.5)
    jOff = int((ll[1]-lam_x0[1])/step + 0.5)
    
    iOff2 = int((ur[0]-lam_x0[0])/step + 0.5)
    jOff2 = int((ur[1]-lam_x0[1])/step + 0.5)
    
    
    # default background
    m.shadedrelief(scale=1.0)

    # other background maps can be downloaded from
    # http://www.naturalearthdata.com/downloads/10m-raster-data/10m-cross-blend-hypso/
    # m.warpimage(image='C:...PATH.../Maps/HYP_HR/HYP_HR_SR_W_DR.tif')

    m.drawcountries(linewidth=0.4)
    m.drawparallels([20.0,30.0,40.0,42.5,50.0,60.0,65.5,70.0,80.0],labels=[1,0,1,0])
    m.drawmeridians(np.arange(0.,361.,10.),labels=[0,0,0,1])


    # the levels of the plotted isolines
    levels = np.arange(976, 1026, 2)
    cs = m.contour(X, Y, data, levels, latlon=True, linewidths=1.0, colors='black')
    # plot the line labels
    plt.clabel(cs, levels[::2], inline=True, fmt='%d', fontsize=11, colors='black')


    if alpha_channel:
        # define mask
        mask = np.isnan(data)[jOff:jOff2,iOff:iOff2]
        alpha_channel = mask*0.35
        cmap = ListedColormap(['black', 'black'])
        m.imshow(mask, alpha=alpha_channel, cmap=cmap)


    if not scatter_pts is None:
        # xcoor = scatter_pts[:,0]
        # ycoor = scatter_pts[:,1]
        xcoor, ycoor = m(scatter_pts[:,0], scatter_pts[:,1])
        m.scatter(xcoor, ycoor, color='red', s=9, marker='.')
    
    plt.show()


###############################################################################

def image_lat_lon_map(data, x0, step, size, scatter_pts=None, alpha_channel=True,
    line_labels=True):
    """
    Does basically the same as plot_lat_lon_map(), but returns a PIL image instead
    of showing a plot of the result.
    """
    # create mesh grid
    gridX = np.arange(0.0, size[1], 1.0)
    gridX = gridX*step + x0[0]
    # y-axis of grid
    gridY = np.arange(0.0, size[0], 1.0)
    gridY = gridY*step + x0[1]
    # arrays holding x- and y-coordinates of grid points
    X, Y = np.meshgrid(gridX, gridY)


    fig = plt.figure(figsize=(16, 12), edgecolor='w', dpi=72)
    canvas = FigureCanvasAgg(fig)
    
    # resolution applies only for vector graphic; 'c' and 'l' too low
    m = Basemap(projection='cyl', resolution='i',
        llcrnrlat=34.5, urcrnrlat=72, llcrnrlon=-26, urcrnrlon=49)
    
    # default background
    m.shadedrelief(scale=1.0)

    # other background maps can be downloaded from
    # http://www.naturalearthdata.com/downloads/10m-raster-data/10m-cross-blend-hypso/
    # m.warpimage(image='C:...PATH.../Maps/HYP_HR/HYP_HR_SR_W_DR.tif')

    m.drawcountries(linewidth=0.4)
    m.drawparallels([20.0,30.0,40.0,50.0,60.0,70.0,80.0],labels=[1,0,0,0])
    m.drawmeridians(np.arange(0.,361.,10.),labels=[0,0,0,1])

    # the levels of the plotted isolines
    levels = np.arange(976, 1026, 2)
    cs = m.contour(X, Y, data, levels, latlon=True, linewidths=1.0, colors='black')
    if line_labels:
        # plot the line labels
        plt.clabel(cs, levels[::2], inline=True, fmt='%d', fontsize=11, colors='black')


    if alpha_channel:
        # define mask
        mask = np.isnan(data)
        alpha_channel = mask*0.35
        cmap = ListedColormap(['black', 'black'])
        m.imshow(mask, alpha=alpha_channel, cmap=cmap)


    if not scatter_pts is None:
        xcoor = scatter_pts[:,0]
        ycoor = scatter_pts[:,1]
        m.scatter(xcoor, ycoor, color='red', s=9, marker='.')
    
    # create image object from this figure
    # fig.tight_layout()
    canvas.draw()
    
    image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    
    # do not show plots
    plt.close()
    
    return image


###############################################################################
###############################################################################

def plot_Iceland_map(data, x0, step, size, scatter_pts=None, alpha_channel=True,
    line_labels=True):
    """
    Plots an isoline visualization of the specified data field on top of an
    Iceland geography background using a lon-lat coordinate system.
    """
    # create mesh grid
    gridX = np.arange(0.0, size[1], 1.0)
    gridX = gridX*step + x0[0]
    # y-axis of grid
    gridY = np.arange(0.0, size[0], 1.0)
    gridY = gridY*step + x0[1]
    # arrays holding x- and y-coordinates of grid points
    X, Y = np.meshgrid(gridX, gridY)

    # the bounds of the Iceland map to be displayed
    lat0 = 61.625
    lat1 = 69.0625
    lon0 = -24.75
    lon1 = -13.25

    fig = plt.figure(figsize=(5.7, 2.9), edgecolor='w', dpi=150)
    # resolution applies only for vector graphic; 'c' and 'l' too low
    m = Basemap(projection='cyl', resolution='i',
        llcrnrlat=lat0, urcrnrlat=lat1, llcrnrlon=lon0, urcrnrlon=lon1)

    # default background
    m.shadedrelief(scale=1.0)

    # other background maps can be downloaded from
    # http://www.naturalearthdata.com/downloads/10m-raster-data/10m-cross-blend-hypso/
    # m.warpimage(image='C:...PATH.../Maps/HYP_HR/HYP_HR_SR_W_DR.tif')

    m.drawcountries(linewidth=0.4)

    m.drawparallels(np.arange(60.,75.,5.),labels=[1,0,0,0])
    m.drawmeridians(np.arange(-25.,-10.,5.),labels=[0,0,0,1])

    # the levels of the plotted isolines
    levels = np.arange(976, 1026, 2)
    cs = m.contour(X, Y, data, levels, latlon=True, linewidths=1.0, colors='black')
    if line_labels:
        # plot the line labels
        plt.clabel(cs, levels, inline=True, fmt='%d', fontsize=11, colors='black')


    if alpha_channel:
        # find indices of relevant data region
        i0 = int((lon0-x0[0])/step + 0.5)
        i1 = int((lon1-x0[0])/step + 0.5)
        j0 = int((lat0-x0[1])/step + 0.5)
        j1 = int((lat1-x0[1])/step + 0.5)

        # define mask
        mask = np.isnan(data[j0:j1,i0:i1])
        alpha_channel = mask*0.35
        cmap = ListedColormap(['black', 'black'])
        m.imshow(mask, alpha=alpha_channel, cmap=cmap)


    if not scatter_pts is None:
        xcoor = scatter_pts[:,0]
        ycoor = scatter_pts[:,1]
        m.scatter(xcoor, ycoor, color='red', s=9, marker='.')
    
    plt.show()


###############################################################################

def plot_Iceland_station_map(scatter_pts, scatter_values):
    """
    Plots the specified station values of the specified values on top of an
    Iceland geography background using a lon-lat coordinate system.
    """
    # the bounds of the Iceland map to be displayed
    lat0 = 61.625
    lat1 = 69.0625
    lon0 = -24.75
    lon1 = -13.25

    fig = plt.figure(figsize=(5.7, 2.9), edgecolor='w', dpi=150)
    # resolution applies only for vector graphic; 'c' and 'l' too low
    m = Basemap(projection='cyl', resolution='i',
        llcrnrlat=lat0, urcrnrlat=lat1, llcrnrlon=lon0, urcrnrlon=lon1)

    # default background
    m.shadedrelief(scale=1.0)

    # other background maps can be downloaded from
    # http://www.naturalearthdata.com/downloads/10m-raster-data/10m-cross-blend-hypso/
    # m.warpimage(image='C:...PATH.../Maps/HYP_HR/HYP_HR_SR_W_DR.tif')

    m.drawcountries(linewidth=0.4)

    m.drawparallels(np.arange(60.,75.,5.),labels=[1,0,0,0])
    m.drawmeridians(np.arange(-25.,-10.,5.),labels=[0,0,0,1])

    # print observation points
    xcoor = scatter_pts[:,0]
    ycoor = scatter_pts[:,1]
    m.scatter(xcoor, ycoor, color='red', s=9, marker='.')
    
    # print observation values in scatter plot - correct label positions if necessary
    for k in range(len(xcoor)):
        if k in [0, 11]:
            i,j = m(xcoor[k]-1.2, ycoor[k]-0.5)        
        elif k in [3, 14]:
            i,j = m(xcoor[k]-0.7, ycoor[k]-0.5)
        elif k in [9]:
            i,j = m(xcoor[k]-0.95, ycoor[k]-0.5)
        else:
            i,j = m(xcoor[k]-0.15, ycoor[k]+0.15)
        plt.annotate(str(scatter_values[k]), (i,j))

    plt.show()


###############################################################################
