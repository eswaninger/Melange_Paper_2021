#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
<<<<<<< HEAD
Created on Wed Mar 10 11:30:24 2021

@author: eswaninger

This script evaluates and plotsglacier speeds 
 
"""
#%% Import Modules
=======
Created on Sun Dec  1 17:08:31 2019

@author: eswaninger

"""
#%%

''' This script gathers glacier speeds 
'''

#%% Modules
>>>>>>> 2f4f0f9f396a45255015b56b0c82b37d044ad2b1

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import datetime as dt
from datetime import date
from datetime import timedelta as td
import math
import pandas as pd
from pylab import *
import statistics as st
import skimage as ski

<<<<<<< HEAD

ice0=[]                               #lists for velocities on glacier
=======
ice0=[]
>>>>>>> 2f4f0f9f396a45255015b56b0c82b37d044ad2b1
ice1=[]
ice2=[]
ice3=[]
ice4=[]
ice5=[]
ice6=[]
ice7=[]
ice8=[]
<<<<<<< HEAD
mel=[]  
=======
melange = []
>>>>>>> 2f4f0f9f396a45255015b56b0c82b37d044ad2b1
block=[]
flist= []
d_list = []
t_start = []
t_end = []
t_diff = []
slopes=[]
medians = []
q1 =[]
q2 =[]
q3 =[]
iqr=[]

yr = 778
xr = 1
ys = np.arange(0,1559)
xs = np.arange(0,845)
for y in ys:
    for x in xs:
<<<<<<< HEAD
#        print((y-yr)/(x-xr))
=======
>>>>>>> 2f4f0f9f396a45255015b56b0c82b37d044ad2b1
        xy = math.atan((x-xr)/(y-yr))
        xy = math.degrees(xy)
        slopes.append(np.abs(xy))
slopes = np.reshape(slopes, (1559,845))
gamma = 180 - np.array(slopes)
beta = 202 #degrees from 0 north (),
alpha = 360 - beta - gamma
alpha = np.reshape(alpha, (1559,845))

for root, dirs, files in os.walk('/data/stor/basic_data/tri_data/rink/proc_data/'):
    for name in files:
        if name.endswith(('adf.unw.rec')):
            file_paths= os.path.join(root, name)
            flist.append(file_paths) 
            flist.sort()                                
            if file_paths.startswith(('/data/stor/basic_data/tri_data/rink/proc_data/d0728')):
                with open(file_paths,'rb') as f:
                    temp= f.read()
                    phase= np.fromfile(file_paths, dtype='>f')
                    phase[phase==0] = np.nan
                    phase_rectangle = np.reshape(phase, (1559,845))#[600:630,755:785]
                    vlos = (-0.0175*phase_rectangle)/(4* 3.14159*(2.5/1440))
#                    flow = vlos/(np.cos(np.radians(alpha)))
                    flow = vlos[610:780,80:340]
                    ice0.append(flow[48,111]) #goldenrod
                    ice1.append(flow[53,130]) #limegreen
                    ice2.append(flow[61,145]) #black
                    ice3.append(flow[74,156]) #red  
                    ice4.append(flow[32,117]) #goldenrod 
                    ice5.append(flow[36,133]) #limegreen
                    ice6.append(flow[46,151]) #black
                    ice7.append(flow[59,163]) #red  
                    ice8.append(flow[88,184]) #blue
                    mel.append(flow[99,145])
                    t_start.append(dt.datetime.strptime(name[:15], '%Y%m%d_%H%M%S'))
                    t_end.append(dt.datetime.strptime(name[17:32], '%Y%m%d_%H%M%S'))  
                    rockfav = np.array(vlos[860:880,380:400])
                    unravel = rockfav.ravel()
                    median = st.median(unravel)
                    medians.append(median)
                    q1.append(np.percentile(unravel, 25, interpolation = 'midpoint'))
                    q2.append(np.percentile(unravel, 50, interpolation = 'midpoint'))
                    q3.append(np.percentile(unravel, 75, interpolation = 'midpoint'))
                    IQR = np.percentile(unravel, 75, interpolation = 'midpoint') - np.percentile(unravel, 25, interpolation = 'midpoint')
                    iqr.append(IQR) 
    
<<<<<<< HEAD
#%%
plt.figure()
#plt.plot(t_start, ice0, '.', color = 'goldenrod')
#plt.plot(t_start, ice1, '.', color = 'limegreen')
plt.plot(t_start, ice2, '.', color = 'black')
plt.plot(t_start, ice3, '.', color = 'red')
#plt.plot(t_start, ice4, '.', color = 'goldenrod')
#plt.plot(t_start, ice5, '.', color = 'limegreen')
plt.plot(t_start, ice6, '.', color = 'black')
#plt.plot(t_start, ice7, '.', color = 'red')
#plt.plot(t_start, ice8, '.', color = 'blue')
plt.axvline(pd.to_datetime('2014-07-29-02:52:00'), color='k', linestyle='--', linewidth = 3, label = 'Calving Event')
plt.axvline(pd.to_datetime('2014-07-28-18:02:00'), color='k', linestyle='--', linewidth = 3)
                    
                    
#%%
file = '20140729_060000u_20140729_060230u.adf.unw.rec'
path= '/data/stor/basic_data/tri_data/rink/proc_data/d0728/INT/REC/'
file_path = (path + file)

with open(file_path,'rb') as f:
    temp= f.read()
    phase= np.fromfile(file_path, dtype='>f')
    phase[phase==0] = np.nan
    phase_rectangle = np.reshape(phase, (1559,845))
    vlos = (-0.0175*phase_rectangle)/(4* 3.14159*(2.5/1440))
    flow = vlos/(np.cos(np.radians(alpha)))
    flow = flow[610:780,80:340]
                   
plt.figure()
#plt.imshow(flow, norm= colors.SymLogNorm(linthresh = np.amin(0.1),linscale=0.19, vmin=9.0, vmax=20.0),cmap = 'viridis')
plt.imshow(flow, vmin = 0, vmax = 20, cmap = 'viridis')
plt.grid()
plt.colorbar()               

#plt.plot([49],[21],'v', color='red', markersize = 6, markerfacecolor= 'none')
#plt.plot([73],[35],'v', color='orangered', markersize = 6, markerfacecolor= 'none')   
#plt.plot([94],[38],'v', color='gold', markersize = 6, markerfacecolor= 'none')
plt.plot([111],[48],'v', color='goldenrod', markersize = 6)
plt.plot([130],[53],'v', color='limegreen', markersize = 6)
#plt.plot([137],[55],'v', color='blue', markersize = 6, markerfacecolor= 'none')
plt.plot([145],[61],'v', color='black', markersize = 6)
plt.plot([156],[74],'v', color='red', markersize = 6)
#plt.plot([164],[77],'v', color='brown', markersize = 6, markerfacecolor= 'none')
#plt.plot([178],[94],'v', color='black', markersize = 6, markerfacecolor= 'none')

#plt.plot([53],[12],'+', color='red', markersize = 6, markerfacecolor= 'none')
#plt.plot([81],[20],'+', color='orangered', markersize = 6, markerfacecolor= 'none')   
#plt.plot([101],[25],'+', color='gold', markersize = 6, markerfacecolor= 'none')
plt.plot([117],[32],'s', color='goldenrod', markersize = 6)
plt.plot([133],[36],'s', color='limegreen', markersize = 6)
#plt.plot([148],[46],'s', color='blue', markersize = 6, markerfacecolor= 'none')
plt.plot([151],[46],'s', color='black', markersize = 6)
plt.plot([163],[59],'s', color='red', markersize = 6)
#plt.plot([171],[68],'+', color='brown', markersize = 6, markerfacecolor= 'none')
plt.plot([184],[88],'s', color='blue', markersize = 6)
#plt.plot([176],[78],'s', color='slateblue', markersize = 6, markerfacecolor= 'none')
#plt.plot([169],[90],'+', color='red', markersize = 6, markerfacecolor= 'none')

plt.plot([145],[99],'s', color='blue', markersize = 6)

#%% Create timeseries data - 3rd calving event 
=======
                    



#%% Create timeseries velocity data
>>>>>>> 2f4f0f9f396a45255015b56b0c82b37d044ad2b1

ice0_vel_meas = np.array(ice0)
ice1_vel_meas = np.array(ice1)
ice2_vel_meas = np.array(ice2)
ice3_vel_meas = np.array(ice3)
ice4_vel_meas = np.array(ice4)
ice5_vel_meas = np.array(ice5)
ice6_vel_meas = np.array(ice6)
ice7_vel_meas = np.array(ice7)
ice8_vel_meas = np.array(ice8)


noise_rand = np.array(iqr)
noise_syst = np.array(medians)

#Dt = 2.5 # minutes
#dt_days = Dt/1440 # days
#t = np.linspace(2, 2+len(gla1_vel_meas)*dt_days, len(gla1_vel_meas))
#t[3251:] += .01 # Add time to the end of the record, to simulate a gap in recording
##x = np.random.normal(loc=noise_syst, scale=noise_rand,
##                     size=[4, 30])
t = t_start
for i in range(len(t_start)):
    t[i] = np.datetime64(t_start[i])
t = np.array(t)

#t = t_start

#plt.plot(t,gla1_vel_meas, '.')
# Correct the measurement data with the systematic noise, as in doing the atmospheric correction
gla1_vel_corr = ice0_vel_meas - noise_syst
gla2_vel_corr = ice1_vel_meas - noise_syst
gla3_vel_corr = ice2_vel_meas - noise_syst
gla4_vel_corr = ice3_vel_meas - noise_syst
gla5_vel_corr = ice4_vel_meas - noise_syst
gla6_vel_corr = ice5_vel_meas - noise_syst
gla7_vel_corr = ice6_vel_meas - noise_syst
gla8_vel_corr = ice7_vel_meas - noise_syst
gla9_vel_corr = ice8_vel_meas - noise_syst


#DEFINE FUNCTIONS
# There shouldn't be a need, I don't think, to modify these functions

# Running Average function with uncertainties for non-evenly spaced data
def running_ave(t, t_span, x, s):
    j = np.argsort(t)
    t_run = t[j]; x = x[j]; s = s[j] # Sort all values into increasing t
    
    x_run = np.full(t.shape, np.nan)
    s_run = np.full(t.shape, np.nan)
    for i in range(len(t_run)): # Loop over all measurement times
        # Find indices within the moving window
        ind = np.where( np.logical_and(t_run >= t_run[i]-t_span/2, 
                                       t_run <= t_run[i]+t_span/2) )[0]
        
        x_run[i] = np.nanmean(x[ind]) # Assign the running average value to the mean of the appropriate values
        # Sum the random errors in quadrature
        error_quad = np.sqrt( np.nansum( s[ind]**2 ) )
        s_run[i] = error_quad/len(ind) # Identify the mean error, which is the total error in quadrature divided by the number of datapoints

    return t_run, x_run, s_run

# For plotting purposes, add nans into a timeseries, in case there
#    are breaks in a timeseries, we don't want them plotted in lines
def nan_into_gap(t, x, thresh = np.min(np.diff(t))*5 ):
    #Default threshold is five times the minimum timestep
    
    # Find the gaps in the timeseries, as defined by thresh
    gap_inds = np.where(np.diff(t) > thresh)[0]
    
    # For each gap, add a nan in the measurements just after the
    #   last measurement, and just before the next measurement
    for i in range(len(gap_inds)):
        gap = gap_inds[i]
        t = np.insert(t, gap+1, t[gap]+thresh/3)
        t = np.insert(t, gap+2, t[gap+2]-thresh/3)
        x = np.insert(x, gap+1, np.nan)
        x = np.insert(x, gap+2, np.nan)
        gap_inds[i+1:] = gap_inds[i+1:]+2
    return t, x

# Calculate Running Averages, add nans to discontinuous data, and plot

# Smooth the timeseries using a running average, with a smoothing 
#   window of t_span days
t_span = np.timedelta64(60, 'm') #2/24 # days
t_run, vel_run_gla1, noise_run = running_ave(t, t_span, gla1_vel_corr, noise_rand)
t_run, vel_run_gla2, noise_run = running_ave(t, t_span, gla2_vel_corr, noise_rand)
t_run, vel_run_gla3, noise_run = running_ave(t, t_span, gla3_vel_corr, noise_rand)
t_run, vel_run_gla4, noise_run = running_ave(t, t_span, gla4_vel_corr, noise_rand)
t_run, vel_run_gla5, noise_run = running_ave(t, t_span, gla5_vel_corr, noise_rand)
t_run, vel_run_gla6, noise_run = running_ave(t, t_span, gla6_vel_corr, noise_rand)
t_run, vel_run_gla7, noise_run = running_ave(t, t_span, gla7_vel_corr, noise_rand)
t_run, vel_run_gla8, noise_run = running_ave(t, t_span, gla8_vel_corr, noise_rand)
t_run, vel_run_gla9, noise_run = running_ave(t, t_span, gla9_vel_corr, noise_rand)


# Add in nans to gappy data smoothed velocity, and gappy smoothed uncertainty,
#    for plotting
gap_thresh = np.timedelta64(30, 'm') #0.01 # days
t_gap, vel_run_gap_g1 = nan_into_gap(t_run, vel_run_gla1, gap_thresh)
t_gap, vel_run_gap_g2 = nan_into_gap(t_run, vel_run_gla2, gap_thresh)
t_gap, vel_run_gap_g3 = nan_into_gap(t_run, vel_run_gla3, gap_thresh)
t_gap, vel_run_gap_g4 = nan_into_gap(t_run, vel_run_gla4, gap_thresh)
t_gap, vel_run_gap_g5 = nan_into_gap(t_run, vel_run_gla5, gap_thresh)
t_gap, vel_run_gap_g6 = nan_into_gap(t_run, vel_run_gla6, gap_thresh)
t_gap, vel_run_gap_g7 = nan_into_gap(t_run, vel_run_gla7, gap_thresh)
t_gap, vel_run_gap_g8 = nan_into_gap(t_run, vel_run_gla8, gap_thresh)
t_gap, vel_run_gap_g9 = nan_into_gap(t_run, vel_run_gla9, gap_thresh)
t_gap, noise_run_gap = nan_into_gap(t_run, noise_run, gap_thresh)    

#
# Plot the temporally-smoothed data Glacier ( in one graph)
plt.figure()

plt.fill_between(t_gap, vel_run_gap_g1-noise_run_gap, vel_run_gap_g1+noise_run_gap, 
                   color='goldenrod', alpha='.2')
#plt.scatter(t, gla1_vel_corr, s=1, color='goldenrod')
plt.plot(t_gap, vel_run_gap_g1, color='goldenrod', markersize = 0.5)

plt.fill_between(t_gap, vel_run_gap_g2-noise_run_gap, vel_run_gap_g2+noise_run_gap, 
                   color='limegreen', alpha='.2')
#plt.scatter(t, gla2_vel_corr, s=1, color='blue')
plt.plot(t_gap, vel_run_gap_g2, color='limegreen', markersize =0.5)

plt.fill_between(t_gap, vel_run_gap_g3-noise_run_gap, vel_run_gap_g3+noise_run_gap, 
                   color='black', alpha='.2')
#plt.scatter(t, gla3_vel_corr, s=1, color='black')
plt.plot(t_gap, vel_run_gap_g3, color='black', markersize =0.5)

plt.fill_between(t_gap[0:320], vel_run_gap_g4[0:320]-noise_run_gap[0:320], vel_run_gap_g4[0:320]+noise_run_gap[0:320], 
                   color='red', alpha='.2')
#plt.scatter(t, gla4_vel_corr[0:320], s=1, color='red')
plt.plot(t_gap[0:320], vel_run_gap_g4[0:320], color='red', markersize =0.5)

plt.fill_between(t_gap, vel_run_gap_g5-noise_run_gap, vel_run_gap_g5+noise_run_gap, 
                   color='goldenrod', alpha='.2')
#plt.scatter(t, gla5_vel_corr, s=1, color='goldenrod')
plt.plot(t_gap, vel_run_gap_g5, '--',color='goldenrod', markersize =0.5)

plt.fill_between(t_gap, vel_run_gap_g6-noise_run_gap, vel_run_gap_g6+noise_run_gap, 
                   color='limegreen', alpha='.2')
#plt.scatter(t, gla6_vel_corr, s=1, color='limegreen')
plt.plot(t_gap, vel_run_gap_g6, '--',color='limegreen', markersize =0.5)

plt.fill_between(t_gap, vel_run_gap_g7-noise_run_gap, vel_run_gap_g7+noise_run_gap, 
                   color='black', alpha='.2')
#plt.scatter(t, gla7_vel_corr, s=1, color='black')
plt.plot(t_gap, vel_run_gap_g7, '--',color='black', markersize =0.5)

plt.fill_between(t_gap, vel_run_gap_g8-noise_run_gap, vel_run_gap_g8+noise_run_gap, 
                   color='red', alpha='.2')
#plt.scatter(t, gla8_vel_corr, s=1, color='red')
plt.plot(t_gap, vel_run_gap_g8, '--',color='red', markersize =0.5)

plt.fill_between(t_gap, vel_run_gap_g9-noise_run_gap, vel_run_gap_g9+noise_run_gap, 
                   color='blue', alpha='.2')
#plt.scatter(t, gla9_vel_corr, s=1, color='blue')
plt.plot(t_gap, vel_run_gap_g9, '--', color='blue', markersize =0.5, label= '▽')

#plt.legend()
plt.xlabel('Time (d)', fontsize = 18)
plt.ylabel('Speeds, with uncertainties (m/d)', fontsize = 18)
plt.title('Velocities (m/d)', fontsize = 24)
#plt.axvspan(pd.to_datetime('2014-07-25-12:45:00'),pd.to_datetime('2014-07-25-15:15:00'), alpha= 0.4, color = 'darkgray', label = 'TRI motion error')
#plt.axvspan(pd.to_datetime('2014-07-29-05:37:30'),pd.to_datetime('2014-07-30-22:25:00'), alpha= 0.2, color = 'gray', label = 'Mélange')
#plt.axvline(pd.to_datetime('2014-07-26-10:00:00'), color='k', linestyle=':', linewidth = 3, label = 'Calving Event')
plt.axvline(pd.to_datetime('2014-07-29-02:52:00'), color='k', linestyle='--', linewidth = 3, label = 'Calving Event')
plt.axvline(pd.to_datetime('2014-07-28-18:02:00'), color='k', linestyle='--', linewidth = 3)

#Legend
#front_line = mpatches.Patch(color='black', marker = '--', label='Frontline Velocity (m/d)')
#back_line = mpatches.Patch(color='black', marker= '-', label='Backline Velocity (m/d)')
#plt.legend(handles=[front_line, back_line])

#
#plt.axvline(pd.to_datetime('2014-07-26-23:45:00'), color='red', linestyle='-.', label = 'A')
#plt.axvline(pd.to_datetime('2014-07-27-09:02:30'), color='orangered', linestyle='-.', label = 'B')
#plt.axvline(pd.to_datetime('2014-07-27-17:25:00'), color='orange', linestyle='-.', label = 'C')
#plt.axvline(pd.to_datetime('2014-07-28-18:02:00'), color='gold', linestyle='-.', label = 'D')
#plt.axvline(pd.to_datetime('2014-07-29-02:15:00'), color='green', linestyle='-.', label = 'E')
#plt.axvline(pd.to_datetime('2014-07-29-04:00:00'), color='blue', linestyle='-.', label = 'F')

plt.legend(fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)   
plt.grid()
#%%
plt.figure()
plt.plot(t_start, gla1_vel_corr, '.', color = 'goldenrod')
plt.plot(t_start, gla2_vel_corr, '.', color = 'limegreen')
plt.plot(t_start, gla3_vel_corr, '.', color = 'black')
plt.plot(t_start, gla4_vel_corr, '.', color = 'red')
plt.plot(t_start, gla5_vel_corr, '.', color = 'goldenrod')
plt.plot(t_start, gla6_vel_corr, '.', color = 'limegreen')
plt.plot(t_start, gla7_vel_corr, '.', color = 'black')
plt.plot(t_start, gla8_vel_corr, '.', color = 'red')
plt.plot(t_start, gla9_vel_corr, '.', color = 'blue')
plt.axvline(pd.to_datetime('2014-07-29-02:52:00'), color='k', linestyle='--', linewidth = 3, label = 'Calving Event')
plt.axvline(pd.to_datetime('2014-07-28-18:02:00'), color='k', linestyle='--', linewidth = 3)

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



<<<<<<< HEAD
#KEY REFERNING INFORMATION ABOUT LOOK DIRECTION AND RADAR LOCATION
=======
#KEY REFERING INFORMATION ABOUT LOOK DIRECTION AND RADAR LOCATION
>>>>>>> 2f4f0f9f396a45255015b56b0c82b37d044ad2b1
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

#%%#%% Same as above but with georeferenced space
adf_size =(1559,845)
pix_spacing = 20 # m  from par file
# READ IN THE MLI
adf_rect_dir = '/data/stor/basic_data/tri_data/rink/proc_data/d0728/INT/REC/'
adf_image = '20140729_030000u_20140729_030230u.adf.unw.rec'
adf_file = (adf_rect_dir + adf_image)

with open(adf_file, mode='rb') as file:
    phase= np.fromfile(adf_file, dtype='>f')
    phase[phase==0] = np.nan
    phase_rectangle = np.reshape(phase, (1559,845))
    adf = (-0.0175*phase_rectangle)/(4* 3.14159*(2.5/1440))
    flows = adf/(np.cos(np.radians(alpha)))
    adf = np.flipud(flows)

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
adf[int(adf_size[0]/2)-1 : int(adf_size[0]/2)+1, 0:2] = radar_loc_tag # "Color" some of the pixels, so that radar location can be found after rotation

# Rotate the image
adf_rot = ski.transform.rotate(adf, rot_angle, resize=True, 
                               center=(0, 5), mode='constant', 
                               cval=-999, preserve_range=True)
adf_rot[np.where(adf_rot==-999)] = np.nan
#mli_rot = imutils.rotate_bound(mli.astype(np.uint8), 0)#rot_angle) # This version required ints, and therefore loss of resolution in the image

# Find the location of the radar within the pixel coordinates
#y_ind, x_ind = np.where(adf_rot == np.array(radar_loc_tag, dtype='float32'))
y_ind, x_ind = [1127],[466]
radar_pix = (max(y_ind), min(x_ind)) # (y, x)   Location in the rotated image where the radar is found.
                                       # Only works for clockwise rotations, where the radar is in top left of colored pixels
#mli_rot[radar_pix[0]:radar_pix[0]+5, radar_pix[1]:radar_pix[1]+5] = np.nan

# Pixel coordinates for image
eas_pix1 = np.arange(adf_rot.shape[1]) - radar_pix[1]
nor_pix1 = np.arange(adf_rot.shape[0]) - radar_pix[0]

# Polar Stereographic coordinates easting and northing
eas1 = (eas_pix1 * pix_spacing) + rad_eas
nor1 = (nor_pix1 * pix_spacing) + rad_nor

e_eas1 = eas1[600:770]
n_nor1 = nor1[930:1150]
#
adf_rot = adf_rot[930:1150, 600:770]

# PLOTTING
plt.figure(num = 1, clear=True)
plt.pcolormesh(e_eas1/1000, n_nor1/1000, adf_rot, vmin = 0, vmax = 20, cmap = 'viridis')
#plt.pcolormesh(eas1/1000, nor1/1000, adf_rot, vmin = 0, vmax = 15, cmap = 'viridis')

plt.colorbar()
plt.axis('equal')
plt.grid(True)
plt.xlabel('Easting (km)', fontsize = 16)
plt.ylabel('Northing (km)', fontsize = 16)


#%% SUBPLOTS of speeds and locations
plt.figure()
axes = plt.subplot2grid((2,2),(1,0), colspan = 2)
#axes0 = plt.subplot2grid((3,3),(2,0), colspan = 2)
axes1 = plt.subplot2grid((2,2), (0,0), colspan=1)
axes2 = plt.subplot2grid((2,2), (0,1), colspan=1)


file = '20140729_041500u_20140729_041730u.adf.unw.rec'
path= '/data/stor/basic_data/tri_data/rink/proc_data/d0728/INT/REC/'
file_path = (path + file)

with open(file_path,'rb') as f:
    temp= f.read()
    phase= np.fromfile(file_paths, dtype='>f')
    phase[phase==0] = np.nan
    phase_rectangle = np.reshape(phase, (1559,845))
    vlos = (-0.0175*phase_rectangle)/(4* 3.14159*(2.5/1440))
    flow = vlos/(np.cos(np.radians(alpha)))
    flow = flow[610:780,80:340]


axes.fill_between(t_gap, vel_run_gap_g1-noise_run_gap, vel_run_gap_g1+noise_run_gap, 
                   color='goldenrod', alpha='.2')
#plt.scatter(t, gla1_vel_corr, s=1, color='darkcyan')
axes.plot(t_gap, vel_run_gap_g1, color='goldenrod', markersize = 0.5)

axes.fill_between(t_gap, vel_run_gap_g2-noise_run_gap, vel_run_gap_g2+noise_run_gap, 
                   color='limegreen', alpha='.2')
#plt.scatter(t, gla2_vel_corr, s=1, color='blue')
axes.plot(t_gap, vel_run_gap_g2, color='limegreen', markersize =0.5)

axes.fill_between(t_gap, vel_run_gap_g3-noise_run_gap, vel_run_gap_g3+noise_run_gap, 
                   color='black', alpha='.2')
#plt.scatter(t, gla3_vel_corr, s=1, color='violet')
axes.plot(t_gap, vel_run_gap_g3, color='black', markersize =0.5, label = '▼')

axes.fill_between(t_gap[0:319], vel_run_gap_g4[0:319]-noise_run_gap[0:319], vel_run_gap_g4[0:319]+noise_run_gap[0:319], 
                   color='red', alpha='.2')
#plt.scatter(t, gla4_vel_corr, s=1, color='magenta')
axes.plot(t_gap[0:319], vel_run_gap_g4[0:319], color='red', markersize =0.5)

axes.fill_between(t_gap, vel_run_gap_g5-noise_run_gap, vel_run_gap_g5+noise_run_gap, 
                   color='goldenrod', alpha='.2')
#plt.scatter(t, gla5_vel_corr, s=1, color='magenta')
axes.plot(t_gap, vel_run_gap_g5,'--', color='goldenrod', markersize =0.5)

axes.fill_between(t_gap, vel_run_gap_g6-noise_run_gap, vel_run_gap_g6+noise_run_gap, 
                   color='limegreen', alpha='.2')
#plt.scatter(t, gla6_vel_corr, s=1, color='violet')
axes.plot(t_gap, vel_run_gap_g6,'--', color='limegreen', markersize =0.5)

axes.fill_between(t_gap, vel_run_gap_g7-noise_run_gap, vel_run_gap_g7+noise_run_gap, 
                   color='black', alpha='.2')
#plt.scatter(t, gla7_vel_corr, s=1, color='magenta')
axes.plot(t_gap, vel_run_gap_g7,'--', color='black', markersize =0.5, label = '■')

axes.fill_between(t_gap, vel_run_gap_g8-noise_run_gap, vel_run_gap_g8+noise_run_gap, 
                   color='red', alpha='.2')
#plt.scatter(t, gla8_vel_corr, s=1, color='magenta')
axes.plot(t_gap, vel_run_gap_g8,'--', color='red', markersize =0.5)

axes.fill_between(t_gap, vel_run_gap_g9-noise_run_gap, vel_run_gap_g9+noise_run_gap, 
                   color='blue', alpha='.2')
#plt.scatter(t, gla9_vel_corr, s=1, color='magenta')
axes.plot(t_gap, vel_run_gap_g9, '--', color='blue', markersize =0.5)


axes.set_xlabel('Date', fontsize = 14)
axes.set_ylabel('Speeds, w/ Uncertainties (m/d)', fontsize = 14)
axes.tick_params(axis='both', which='major', labelsize=12)
axes.grid()
#axes.set_title('Glacier Velocities (m/d)', fontsize = 15)
axes.axvline(pd.to_datetime('2014-07-28-18:02:00'), color='k', linestyle=':', linewidth = 3, label = 'Calving Event')
axes.axvline(pd.to_datetime('2014-07-29-02:52:00'), color='k', linestyle=':', linewidth = 3)
axes.axvspan(pd.to_datetime('2014-07-29-04:00:00'),pd.to_datetime('2014-07-29-19:25:00'), alpha= 0.2, color = 'gray', label = 'Mélange')
axes.legend(loc= 'best')


fig = axes1.pcolormesh(e_eas1/1000, n_nor1/1000, adf_rot, vmin =0, vmax = 20, cmap = 'viridis') #norm=colors.SymLogNorm(linthresh = np.amin(5),linscale=0.090, vmin=0, vmax=16.5)
#axes1.axis('equal')
#axes1.set_title('20140729_031500.adf.unw.rec')
axes1.set_xlabel('Easting (km)', fontsize = 14)
axes1.set_ylabel('Northing (km)', fontsize = 14)

axes1.grid(True)
axes1.axis('equal')
#axes1.yaxis.set_major_locator(plt.NullLocator())
#axes1.xaxis.set_major_formatter(plt.NullFormatter()) 
plt.colorbar(fig,ax= axes1, label ='Speeds (m/d)')  

axes1.plot([-231.257],[-1980.45],'v', color='goldenrod', markersize = 5)
axes1.plot([-230.885],[-1980.39],'s', color='goldenrod', markersize = 5)
axes1.plot([-231.123],[-1980.67],'v', color='limegreen', markersize = 5)
axes1.plot([-230.809],[-1980.62],'s', color='limegreen', markersize = 5)
axes1.plot([-230.908],[-1981.07],'v', color='black', markersize = 5)
axes1.plot([-230.66],[-1981.07],'s', color='black', markersize = 5)
axes1.plot([-230.613],[-1981.32],'s', color='red', markersize = 5)
axes1.plot([-230.962],[-1981.38],'v', color='red', markersize = 5)
axes1.plot([-230.649],[-1981.96],'s', color='blue', markersize = 5)



<<<<<<< HEAD
#pos0 = axes.get_position()
#pos1 = [pos0.x0 + 0.3, pos0.y0 + 0.3,  pos0.width / 2.0, pos0.height / 2.0] 
#axes1.set_position(pos1)

=======
>>>>>>> 2f4f0f9f396a45255015b56b0c82b37d044ad2b1
axes2.pcolormesh(e_eas/1000, n_nor/1000, mli_rot, vmin=0, vmax=16, cmap = 'gray')
axes2.axis('equal')
axes2.grid(True)
#axes2.set_title('20140729_105000u.mli.rec')
axes2.set_xlabel('Easting (km)', fontsize = 14)
#axes2.set_ylabel('Northing (km)', fontsize = 14)
axes2.plot([-232.909,-232.845,-232.359,-231.76,-231.71,-231.485,-231.499,-231.393,
          -231.24,-231.176,-231.173,-230.979,-230.908,-230.659],
         [-1979.67,-1979.78,-1979.83,-1980.38,-1980.7,-1981.44,-1981.98,-1982.3,-1982.46,
          -1982.75,-1983.08,-1982.95,-1983.32,-1983.6],
         color='limegreen', markersize = 1.5)   

axes2.plot([-232.909,-232.7,-232.41,-231.925,-231.533,-231.467,-231.415,
            -231.217,-231.223,-231.068,-230.979,-230.886,-230.735,-230.543],
         [-1979.67,-1979.95,-1979.86,-1980.1,-1980.47,-1980.72,-1981.4,
          -1981.83, -1982.20,-1982.42,-1983.07,-1983.39,-1983.37,-1983.62],
         color='yellow', markersize = 1.5) 
#
axes2.plot([-232.911,-232.709,-232.314,-231.807,-231.672,-231.287,-231.137,
            -230.988,-230.977,-230.896,-230.931,-230.815,-230.636,-230.571],
         [-1979.67,-1979.91,-1979.86,-1980.15,-1980.16,-1980.59,-1981.07,
          -1981.55, -1982.30,-1982.64,-1982.89,-1983.36,-1983.4,-1983.65],
         color='orange', markersize = 1.5) 

axes2.plot([-232.891,-232.786,-232.685,-232.295,-231.831,-231.477,-231.267,
            -231.163,-230.935,-230.887,-230.689,-230.584,-230.719,-230.638,
            -230.791,-230.713,-230.587,-230.545],
         [-1979.64,-1979.86,-1979.93,-1979.84,-1980.12,-1980.30,-1980.62,
          -1980.96, -1981.17,-1981.29,-1981.19,-1981.38,-1982.66,-1982.86,
          -1983.04,-1983.42,-1983.41,-1983.64],
         color='red', markersize = 1.5)  
axes2.plot([-231.235],[-1980.36],'v', color='goldenrod', markersize = 5)
axes2.plot([-230.901],[-1980.22],'s', color='goldenrod', markersize = 5)
axes2.plot([-231.169],[-1980.74],'v', color='limegreen', markersize = 5)
axes2.plot([-230.763],[-1980.48],'s', color='limegreen', markersize = 5)
axes2.plot([-230.832],[-1981.11],'v', color='white', markersize = 5, markerfacecolor= 'none')
axes2.plot([-230.56],[-1981.02],'s', color='white', markersize = 5, markerfacecolor= 'none')
axes2.plot([-230.892],[-1981.47],'v', color='red', markersize = 5)
axes2.plot([-230.578],[-1981.25],'s', color='red', markersize = 5)
axes2.plot([-230.52],[-1981.89],'s', color='blue', markersize = 5)


axes1.yaxis.set_label_coords(-0.14,0.5)
axes.yaxis.set_label_coords(-0.05,0.5)
plt.setp(axes.get_xticklabels(),visible=True)
<<<<<<< HEAD
#fig.align_ylabels(axes[:, 1])
=======
#fig.align_ylabels(axes[:, 1])
>>>>>>> 2f4f0f9f396a45255015b56b0c82b37d044ad2b1
