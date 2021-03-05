#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:32:26 2019

@author: eswaninger
"""

#%% Modules

import matplotlib.pyplot as plt
import numpy as np
import Cython as c
import setuptools as sts
import cftime as cf
from netCDF4 import Dataset
import netCDF4 as nc
import os

#%%
#%% Bedmap files at Rink

os.chdir('/data/stor/basic_data/spatial_data/greenland/bedmap/')
filename = 'BedMachineGreenland-2017-09-20.nc'

file = nc.Dataset(filename)

ice_thickness = file['thickness'] #meters
bed_elevation = file['bed'] #meters
mask = file['mask'] #0 = ocean, 1 = ice-free land, 2 = grounded ice, 3 = floating ice, 4 = non-Greenland land
sur = file['surface']

plt.figure()
plt.imshow(ice_thickness[8875:9050,2750:2925]) #[2813.83,8986.85]
plt.figure()
plt.imshow(bed_elevation[8875:9050, 2750:2925])
plt.figure()
plt.imshow(sur[8875:9050, 2750:2925])
plt.figure()
plt.imshow(mask[8875:9050,2750:2925])

plt.imshow(mask)
plt.figure()
plt.imshow(ice_thickness)
plt.figure()
plt.imshow(bed_elevation)

plt.plot([62.54],[115.2],'o', color='red', markersize = 6)  

#%% Subplots of all
plt.figure()
ax0 = plt.subplot2grid((2,2),(0,0), colspan = 1)
ax1 = plt.subplot2grid((2,2), (1,0), colspan=1)
ax2= plt.subplot2grid((2,2), (0,1), colspan=1)
ax3= plt.subplot2grid((2,2), (1,1), colspan=1)

ax0.imshow(ice_thickness[8875:9050,2750:2925], vmin = 0, vmax = 1000) 
ax0.set_title('ice thickness (m)')
ax0.plot([64.54],[120.2],'o', color='red', markersize = 6)  
ax1.imshow(bed_elevation[8875:9050, 2750:2925], vmin = -1000, vmax = 1000)
ax1.set_title('bed elevation (m)')
ax1.plot([64.54],[120.2],'o', color='red', markersize = 6)  
ax2.imshow(sur[8875:9050, 2750:2925], vmin = 0, vmax = 1000)
ax2.set_title('surface elevation (m)')
ax2.plot([64.54],[120.2],'o', color='red', markersize = 6)  
ax3.imshow(mask[8875:9050,2750:2925])
ax3.set_title('ice vs. not ice')
ax3.plot([64.54],[120.2],'o', color='red', markersize = 6)  













