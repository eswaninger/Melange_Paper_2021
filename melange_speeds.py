#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:12:13 2020

@author: eswaninger
"""

#%%
''' This script looks at speeds at two positions within the melange to observe
the rigidity and motion
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os 
import math
import datetime as dt
from datetime import timedelta as dt
import datetime as dt
import statistics as st
import pandas as pd
import skimage as ski

#%%
mlist= []
disspot= [] 
gla_1=[]
gla_2=[]
gla_3=[]
mel_1=[]
mel_2=[]
t_start = []
t_end = []
flist= []
med=[]
medians=[] #systematic noise
means=[]
std=[]      #Random Noise
vari=[]
q1=[]
q2=[]
q3=[]
iqr = []

for root, dirs, files in os.walk('/data/stor/basic_data/tri_data/rink/proc_data/'):
    for name in sorted(files):
        if name.endswith(('.adf.unw')):
            file_paths= os.path.join(root, name)
            flist.append(file_paths) 
            allfiles= np.array(flist)
            if file_paths.startswith(('/data/stor/basic_data/tri_data/rink/proc_data/d0727')):
                with open(file_paths,'rb') as f:
                        temp= f.read()
                        vel_= np.fromfile(file_paths, dtype='>f')
                        vel_rectangle= np.reshape(vel_, (674,1495))#[600:630,755:785]
                        vel_[vel_==0] = np.nan
                        rotvel = np.rot90(vel_rectangle) #These are the rotated velocities             
                        mel = rotvel                                 # LOS melange velocities
                        mel_1.append(mel[1057, 257])
                        mel_2.append(mel[1176, 317]) 
                        t_start.append(dt.datetime.strptime(name[:15], '%Y%m%d_%H%M%S'))
                        t_end.append(dt.datetime.strptime(name[17:32], '%Y%m%d_%H%M%S'))  
                        rockfav = np.array(rotvel[815:835,340:360])
                        unravel = rockfav.ravel()
                        median = st.median(unravel)
                        medians.append(median)
                        q1.append(np.percentile(unravel, 25, interpolation = 'midpoint'))
                        q2.append(np.percentile(unravel, 50, interpolation = 'midpoint'))
                        q3.append(np.percentile(unravel, 75, interpolation = 'midpoint'))
                        IQR = np.percentile(unravel, 75, interpolation = 'midpoint') - np.percentile(unravel, 25, interpolation = 'midpoint')
                        iqr.append(IQR)

            if file_paths.startswith(('/data/stor/basic_data/tri_data/rink/proc_data/d0728')):
                with open(file_paths,'rb') as f:
                        temp= f.read()
                        vel_= np.fromfile(file_paths, dtype='>f')
                        vel_rectangle= np.reshape(vel_, (674,1495))#[600:630,755:785]
                        vel_[vel_==0] = np.nan
                        rotvel = np.rot90(vel_rectangle) #These are the rotated velocities             
                        mel = rotvel                                 # LOS melange velocities
                        mel_1.append(mel[1057, 257])
                        mel_2.append(mel[1176, 317]) 
                        t_start.append(dt.datetime.strptime(name[:15], '%Y%m%d_%H%M%S'))
                        t_end.append(dt.datetime.strptime(name[17:32], '%Y%m%d_%H%M%S'))  
                        rockfav = np.array(rotvel[815:835,340:360])
                        unravel = rockfav.ravel()
                        median = st.median(unravel)
                        medians.append(median)
                        q1.append(np.percentile(unravel, 25, interpolation = 'midpoint'))
                        q2.append(np.percentile(unravel, 50, interpolation = 'midpoint'))
                        q3.append(np.percentile(unravel, 75, interpolation = 'midpoint'))
                        IQR = np.percentile(unravel, 75, interpolation = 'midpoint') - np.percentile(unravel, 25, interpolation = 'midpoint')
                        iqr.append(IQR)
                        
            if file_paths.startswith(('/data/stor/basic_data/tri_data/rink/proc_data/d0729')):
                with open(file_paths,'rb') as f:
                        temp= f.read()
                        vel_= np.fromfile(file_paths, dtype='>f')
                        vel_rectangle= np.reshape(vel_, (674,1495))#[600:630,755:785]
                        vel_[vel_==0] = np.nan
                        rotvel = np.rot90(vel_rectangle) #These are the rotated velocities             
                        mel = rotvel                                 # LOS melange velocities
                        mel_1.append(mel[1057, 257])
                        mel_2.append(mel[1176, 317])   
                        t_start.append(dt.datetime.strptime(name[:15], '%Y%m%d_%H%M%S'))
                        t_end.append(dt.datetime.strptime(name[17:32], '%Y%m%d_%H%M%S'))  
                        rockfav = np.array(rotvel[815:835,340:360])
                        unravel = rockfav.ravel()
                        median = st.median(unravel)
                        medians.append(median)
                        q1.append(np.percentile(unravel, 25, interpolation = 'midpoint'))
                        q2.append(np.percentile(unravel, 50, interpolation = 'midpoint'))
                        q3.append(np.percentile(unravel, 75, interpolation = 'midpoint'))
                        IQR = np.percentile(unravel, 75, interpolation = 'midpoint') - np.percentile(unravel, 25, interpolation = 'midpoint')
                        iqr.append(IQR)
                        
            if file_paths.startswith(('/data/stor/basic_data/tri_data/rink/proc_data/d0731')):
                with open(file_paths,'rb') as f:
                        temp= f.read()
                        vel_= np.fromfile(file_paths, dtype='>f')
                        vel_rectangle= np.reshape(vel_, (674,1495))#[600:630,755:785]
                        vel_[vel_==0] = np.nan
                        rotvel = np.rot90(vel_rectangle) #These are the rotated velocities             
                        mel = rotvel                                 # LOS melange velocities
                        mel_1.append(mel[1057, 257])
                        mel_2.append(mel[1176, 317]) 
                        t_start.append(dt.datetime.strptime(name[:15], '%Y%m%d_%H%M%S'))
                        t_end.append(dt.datetime.strptime(name[17:32], '%Y%m%d_%H%M%S'))  
                        rockfav = np.array(rotvel[815:835,340:360])
                        unravel = rockfav.ravel()
                        median = st.median(unravel)
                        medians.append(median)
                        q1.append(np.percentile(unravel, 25, interpolation = 'midpoint'))
                        q2.append(np.percentile(unravel, 50, interpolation = 'midpoint'))
                        q3.append(np.percentile(unravel, 75, interpolation = 'midpoint'))
                        IQR = np.percentile(unravel, 75, interpolation = 'midpoint') - np.percentile(unravel, 25, interpolation = 'midpoint')
                        iqr.append(IQR)
#                    
 #%%
mli_size = (1248, 677)
pix_spacing = 25 # m  from par file

#Open MLI image
#rmli_directory25 = '/data/stor/basic_data/tri_data/rink/proc_data/d0725/MLI/'
mli_rect_dir = '/data/stor/basic_data/tri_data/rink/old//MLI_rect/'
mli_image = '20140729_105000u.mli.rec'
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
plt.pcolormesh(e_eas/1000, n_nor/1000, mli_rot, vmin=0, vmax=10, cmap = 'gray')#20)

plt.axis('equal')
plt.grid(True)
plt.xlabel('Easting (km)', fontsize = 14)
plt.ylabel('Northing (km)', fontsize = 14)                
                              
                
                
#%%
#ice_vel_meas = np.array(ice)
mel0_vel_meas = np.array(mel_1)
mel1_vel_meas = np.array(mel_2)



noise_rand = np.array(iqr)
noise_syst = np.array(medians)

#Dt = 2.5 # minutes
#dt_days = Dt/1440 # days
#t = np.linspace(2, 2+len(gla1_vel_meas)*dt_days, len(gla1_vel_meas))
#t[3251:] += .01 # Add time to the end of the record, to simulate a gap in recording
##x = np.random.normal(loc=noise_syst, scale=noise_rand,
##                     size=[4, 30])
t = t_end
for i in range(len(t_start)):
    t[i] = np.datetime64(t_start[i])
t = np.array(t)

#t = t_start

#plt.plot(t,gla1_vel_meas, '.')
# Correct the measurement data with the systematic noise, as in doing the atmospheric correction
#gla_vel_corr = ice_vel_meas - noise_syst
mel0_vel_corr = mel0_vel_meas - noise_syst
mel1_vel_corr = mel1_vel_meas - noise_syst

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
t_span = np.timedelta64(150, 'm') #2/24 # days
#t_run, vel_run_gla, noise_run = running_ave(t, t_span, gla_vel_corr, noise_rand)
t_run, vel_run_mel0, noise_run = running_ave(t, t_span, mel0_vel_corr, noise_rand)
t_run, vel_run_mel1, noise_run = running_ave(t, t_span, mel1_vel_corr, noise_rand)


# Add in nans to gappy data smoothed velocity, and gappy smoothed uncertainty,
#    for plotting
gap_thresh = np.timedelta64(15, 'm') #0.01 # days
#t_gap, vel_run_gap_g = nan_into_gap(t_run, vel_run_gla, gap_thresh)
t_gap, vel_run_gap_m0 = nan_into_gap(t_run, vel_run_mel0, gap_thresh)
t_gap, vel_run_gap_m1 = nan_into_gap(t_run, vel_run_mel1, gap_thresh)
t_gap, noise_run_gap = nan_into_gap(t_run, noise_run, gap_thresh)   
#%%  Plot data, with corrections for systematic error
#
fig, ax = plt.subplots(nrows=4, num=1, sharex=True, clear=True)
ax[0].scatter(t, mel0_vel_meas, label='Measured', marker = '.')
ax[0].scatter(t, mel1_vel_corr, label='Corrected for atmos', marker = '.')
ax[0].set_ylabel('Speed (m/d)')
ax[0].legend()

ax[1].scatter(t, noise_syst, marker = '.')
ax[1].set_ylabel('Systematic \nerror (m/d)')

ax[2].scatter(t, noise_rand, marker = '.')
ax[2].set_ylabel('Random \nerror (m/d)')

#ax[3].fill_between(t, gla1_vel_corr-noise_rand, gla2_vel_corr+noise_rand, 
#                   color='tab:orange', alpha='.2')
ax[3].plot(t, mel1_vel_corr, color='tab:orange', markersize = 0.25)
ax[3].set_ylabel('Corr Speed \nw/ uncertainty (m/d)')

#%%
# Plot the temporally-smoothed data Glacier ( in one graph)
fig, axes = plt.subplots(nrows= 2, ncols= 1, clear = True)


file = '20140729_031500u_20140729_031730u.adf.unw.rec'
path= '/data/stor/basic_data/tri_data/rink/proc_data/d0728/INT/REC/'
file_path = (path + file)

with open(file_path,'rb') as f:
    temp= f.read()
    phase= np.fromfile(file_paths, dtype='>f')
    phase[phase==0] = np.nan
    phase_rectangle = np.reshape(phase, (674,1495))
    vlos = (-0.0175*phase_rectangle)/(4* 3.14159*(2.5/1440))
#    flow = vlos/(np.cos(np.radians(alpha)))
#    flow = flow[610:780,80:340]

axes[1].fill_between(t_gap, vel_run_gap_m0-noise_run_gap, vel_run_gap_m0+noise_run_gap, 
                   color='magenta', alpha='.2')
#plt.scatter(t, gla1_vel_corr, s=1, color='red')
axes[1].plot(t_gap, vel_run_gap_m0, color='magenta', markersize = 0.5)

axes[1].fill_between(t_gap, vel_run_gap_m1-noise_run_gap, vel_run_gap_m1+noise_run_gap, 
                   color='turquoise', alpha='.2')
#plt.scatter(t, gla3_vel_corr, s=1, color='dimgray')
axes[1].plot(t_gap, vel_run_gap_m1, color='turquoise', markersize =0.5)
#axes[1].set_ylim(0,25)

#axes[0].set_xlabel('Time (d)', fontsize = 14)
axes[1].set_ylabel('Line-of-sight speed (m/d)', fontsize = 12)
axes[1].axvspan(pd.to_datetime('2014-07-29-04:00:0'),pd.to_datetime('2014-07-30-20:25:00'), alpha= 0.2, color = 'gray', label = 'Mélange')
axes[1].axvline(pd.to_datetime('2014-07-29-02:52:00'), color='k', linestyle='--', linewidth = 2, label = 'Calving Event')
axes[1].axvline(pd.to_datetime('2014-07-28-18:02:00'), color='k', linestyle='--', linewidth = 2)  
axes[1].legend()
axes[1].grid()  
axes[1].set_xlabel('Time (d)', fontsize = 12)

axes[0].pcolormesh(e_eas/1000, n_nor/1000, mli_rot, vmin=0, vmax=10, cmap = 'gray')#20)
axes[0].axis('equal')
axes[0].grid(True)
axes[0].set_xlabel('Easting (km)', fontsize = 12)
axes[0].set_ylabel('Northing (km)', fontsize = 12)                             
axes[0].plot([-233.021],[-1982.26],'v', color='turquoise', markersize = 6)
axes[0].plot([-231.101],[-1981.41],'v', color='magenta', markersize = 6)    
axes[0].yaxis.set_label_coords(-0.1,0.5)
axes[1].yaxis.set_label_coords(-0.1,0.5)               
#plt.gcf().autofmt_xdate()
#pos0 = axes[0].get_position()
#pos1 = [pos0.x0 + 0.3, pos0.y0 + 0.3,  pos0.width / 2.0, pos0.height / 2.0] 
#axes[1].set_position(pos1)
#%%
 
 
 
 
 
 
 
 
 