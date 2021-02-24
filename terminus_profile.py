#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:39:42 2020

@author: eswaninger
"""
#%% Modules

import numpy as np
import skimage as ski
import matplotlib.pyplot as plt


#%%#%% TERMINUS PROFILES in georeferenced space

# READ IN THE MLI
mli_size = (1248, 677)
pix_spacing = 25 # m  from par file

#Open MLI image
#rmli_directory25 = '/data/stor/basic_data/tri_data/rink/proc_data/d0725/MLI/'
mli_rect_dir = '/data/stor/basic_data/tri_data/rink/old//MLI_rect/'
mli_image = '20140728_235000u.mli.rec'
#mli_image = '20140726_182730u.mli.rec' # first adjusted scan image
#mli_image = '20140728_182730u.mli.rec' # after first calving event
#mli_image = '20140729_033000u.mli.rec' # after second calving event
mli_file = (mli_rect_dir + mli_image)

with open(mli_file, mode='rb') as file:
    mli_array = file.read()[:]
    
pixels = np.fromfile(mli_file, dtype='>f')
pixels[pixels==0] = np.nan
mli = np.reshape(pixels, mli_size) #1118, 1035
mli = np.flipud(mli)
#mli_rot = np.rot90(mli) #These are the rotated velocities



#KEY REFERNING INFORMATION ABOUT LOOK DIRECTION AND RADAR LOCATION
# These locations are for the second TRI placement
# Based on location of radar on Aug 1, 2014, working with field notes and 
    # /data/stor/basic_data/tri_data/rink/matlab_scripts/find_look_angle_and_azimuth/Find_look_angle.m
    # Script written May 7, 2015
rot_angle = 36.7 # %deg clockwise accurate to +/- 1 deg
# From radar_loc = [71.74135 -51.79122]; with polarstereo_fwd
    # In matlab, this is: [rad_eas, rad_nor]=polarstereo_fwd(radar_loc(1), radar_loc(2), 6378137.0, 0.08181919, 70, -45);
rad_nor = -1980152.74511
rad_eas = -235811.17215

# ROTATE AND ORIENT THE MLI

# Tag the location of the radar within the pixel coordinates of the unrotated image.
radar_loc_tag = 999.123
mli[int(mli_size[0]/2)-1 : int(mli_size[0]/2)+1, 0:2] = radar_loc_tag # "Color" some of the pixels, so that radar location can be found after rotation

# Rotate the image
mli_rot = ski.transform.rotate(mli, rot_angle, resize=True, 
                               center=(0, 5), mode='constant', 
                               cval=-999, preserve_range=True)
mli_rot[np.where(mli_rot==-999)] = np.nan
#mli_rot = imutils.rotate_bound(mli.astype(np.uint8), 0)#rot_angle) # This version required ints, and therefore loss of resolution in the image

# Find the location of the radar within the pixel coordinates
y_ind, x_ind = np.where(mli_rot == np.array(radar_loc_tag, dtype='float32'))
radar_pix = (max(y_ind), min(x_ind)) # (y, x)   Location in the rotated image where the radar is found.
                                       # Only works for clockwise rotations, where the radar is in top left of colored pixels

#mli_rot[radar_pix[0]:radar_pix[0]+5, radar_pix[1]:radar_pix[1]+5] = np.nan

# Pixel coordinates for image
eas_pix = np.arange(mli_rot.shape[1]) - radar_pix[1]
nor_pix = np.arange(mli_rot.shape[0]) - radar_pix[0]

# Polar Stereographic coordinates easting and northing
eas = (eas_pix * pix_spacing) + rad_eas
nor = (nor_pix * pix_spacing) + rad_nor

e_eas = eas[450:700]
n_nor = nor[763:984]
#
mli_rot = mli_rot[763:984, 450:700]
# PLOTTING
plt.figure(num = 1, clear=True)
plt.pcolormesh(e_eas/1000, n_nor/1000, mli_rot, vmin=0, vmax=20, cmap = 'gray')#20)

plt.axis('equal')
plt.grid(True)
plt.xlabel('Easting (km)', fontsize = 14)
plt.ylabel('Northing (km)', fontsize = 14)

plt.plot([-232.909,-232.845,-232.359,-231.76,-231.71,-231.485,-231.499,-231.393,
          -231.24,-231.176,-231.173,-230.979,-230.908,-230.659],
         [-1979.67,-1979.78,-1979.83,-1980.38,-1980.7,-1981.44,-1981.98,-1982.3,-1982.46,
          -1982.75,-1983.08,-1982.95,-1983.32,-1983.6],
         color='limegreen', markersize = 1.5)   

plt.plot([-232.909,-232.7,-232.41,-231.925,-231.533,-231.467,-231.415,
            -231.217,-231.223,-231.068,-230.979,-230.886,-230.735,-230.543],
         [-1979.67,-1979.95,-1979.86,-1980.1,-1980.47,-1980.72,-1981.4,
          -1981.83, -1982.20,-1982.42,-1983.07,-1983.39,-1983.37,-1983.62],
         color='yellow', markersize = 1.5) 
#
plt.plot([-232.911,-232.709,-232.314,-231.807,-231.672,-231.287,-231.137,
            -230.988,-230.977,-230.896,-230.931,-230.815,-230.636,-230.571],
         [-1979.67,-1979.91,-1979.86,-1980.15,-1980.16,-1980.59,-1981.07,
          -1981.55, -1982.30,-1982.64,-1982.89,-1983.36,-1983.4,-1983.65],
         color='orange', markersize = 1.5) 

plt.plot([-232.891,-232.786,-232.685,-232.295,-231.831,-231.477,-231.267,
            -231.163,-230.935,-230.887,-230.689,-230.584,-230.719,-230.638,
            -230.791,-230.713,-230.587,-230.545],
         [-1979.64,-1979.86,-1979.93,-1979.84,-1980.12,-1980.30,-1980.62,
          -1980.96, -1981.17,-1981.29,-1981.19,-1981.38,-1982.66,-1982.86,
          -1983.04,-1983.42,-1983.41,-1983.64],
         color='red', markersize = 1.5)  