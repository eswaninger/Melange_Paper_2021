#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:18:34 2019

@author: eswaninger
"""

#%%
''' This script uses openPIV to find the true flow direction of Rink and used 
to monitor movement within melange while it was hardening and moving within 
the fjord.
''' 
#%% Modules

import os
import openpiv
import time
import numpy as np
import argparse
import cv2
import glob
import sys
import scipy
import cython
import datetime as dt
import matplotlib.pyplot as plt
from  matplotlib.pyplot import quiver
from matplotlib.image import imread
import matplotlib.cm as cm
import skimage.io as io
import skimage as ski
import matplotlib.colors as colors
import imutils

from scipy import ndimage
from scipy import optimize
from scipy import signal
from PIL import Image

#openPIV Modules
import openpiv.tools
import openpiv.pyprocess
import openpiv.scaling
import openpiv.validation
import openpiv.filters

#%% openPIV processing an image pair
#change working directory to path below by running line
cd /data/stor/basic_data/tri_data/rink/old/MLI_rect/
cd /data/stor/basic_data/tri_data/rink/proc_data/d0728/MLI

#%% #MLI_BMP FILES WITH PATHS EXAMPLES

files = glob.glob('/data/stor/basic_data/tri_data/rink/old/MLI/*l.mli.bmp')

mli_1 = openpiv.tools.imread('20140729_033730u.mli.rec.bmp') #ice melange starting to harden after second calving event
mli_2 = openpiv.tools.imread('20140729_043730u.mli.rec.bmp') #starting to disintegrate from the ocean inward

#%%
#Read in image
#mli_1 = openpiv.tools.imread('20140729_070000u.mli.rec.bmp') 
#mli_2 = openpiv.tools.imread('20140729_080000u.mli.rec.bmp')

mli_1 = openpiv.tools.imread('20140728_060000u.mli.rec.bmp') 
mli_2 = openpiv.tools.imread('20140729_070000u.mli.rec.bmp')

##VELOCITY FLOW FIELDS PLOTTING


u, v, sig2noise = openpiv.pyprocess.extended_search_area_piv(mli_1, mli_2, window_size=22, overlap=18, dt=0.02, search_area_size=22, sig2noise_method='peak2peak')
# window_size is the size of the interrogation window on mli_1
# overlap is the pixels between adjacent windows
# dt is the the time delay in seconds between the two image frames
# sign2noise_method is the method to use for the evaluation of the signal/noise ratio
# returns 3rd array, sig3noise which is the signal to noise ration obtained from each cross- correlation

x, y = openpiv.pyprocess.get_coordinates(image_size=mli_1.shape, window_size=22, overlap=18)
#compute the coordinates of the centers of the interrogation windows

#plt.quiver(x,y,u,v, color='blue', linewidth = 0.5)

u, v, mask = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = 1.8)
#classifing a vector as an outlier if its signal to noise ratio exceeds a certain threshold
#sets to NaN all those vector for which the signael to noise ratio is below 1.3

#u, v = openpiv.filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=3)

x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = 1)
# uniform scaling to the flow field to get dimensional units

color = np.sqrt(u**2 + v**2)


plt.figure()
plt.quiver(x,y,u,v, color, norm = colors.SymLogNorm(linthresh = np.amin(5),linscale=1, vmin=0, vmax=230), headwidth = 3, headlength = 5, headaxislength = 4) 
#plt.streamplot(x,y,u,v, density = 10) 
#Need to zoom in so the picture doesn't flip to get image in right position
#plt.imshow(np.flip(np.flip(mli_2),axis =1), cmap = 'gray')

#plt.gcf().autofmt_xdate() 
plt.minorticks_on()
plt.axis('equal')
plt.colorbar()
#%% Run code below after you zoom into area you want to observe with the quiver 
#vectors otherwise the magnitude will be flipped
plt.imshow(np.flipud(mli_2), cmap = 'gray')
plt.grid()

#%%
   

