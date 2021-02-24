#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:54:01 2020

@author: eswaninger
"""

#%%
''' This script calculates the displacement experienced by the ice tooth
as it is pulled from the wall with the rotating iceberg that calved during 
the third calving event'''

#%% Modules

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os 
import math
import datetime as dt
from datetime import timedelta as td
import statistics as st
import pandas as pd
import skimage as ski
import scipy

#%%
ice0=[]
ice1=[]
ice0a=[]
ice1a=[]
flist= []
d_list = []
t_start = []
t_end = []
t_diff = []
medians = []
q1 =[]
q2 =[]
q3 =[]
iqr=[]
name_list=[]
slopes=[]

# The lines leading up to the alpha are creation of angles to be used for 
# correcting LOS speed to true flow direction

yr = 1
xr = 1
ys = np.arange(0,1559)
xs = np.arange(0,845)
for y in ys:
    for x in xs:
#        print((y-yr)/(x-xr))
        xy = math.atan((x-xr)/(y-yr))
        xy = math.degrees(xy)
        slopes.append(np.abs(xy))
slopes = np.reshape(slopes, (1559,845))
gamma = 180 - np.array(slopes)
beta = 202 #degrees from 0 north 
alpha = 360 - beta - gamma
alpha = np.reshape(alpha, (1559,845))

#Looping through directories to get speeds on 28th and 29th for 
#2nd and 3rd calving events and melange development
#not using rect files since d0729 does not have rect files
for root, dirs, files in os.walk('/data/stor/basic_data/tri_data/rink/proc_data/'):
    for name in sorted(files):
        if name.endswith(('adf.unw.rec')):
            file_paths= os.path.join(root, name)
            flist.append(file_paths) 
#            flist.sort()   
            if file_paths.startswith(('/data/stor/basic_data/tri_data/rink/proc_data/d0728/INT/')):
                with open(file_paths,'rb') as f:
                    temp= f.read()
                    pha_= np.fromfile(file_paths, dtype='>f')
                    pha_[pha_==0] = np.nan
                    pha_rectangle = np.reshape(pha_, (1559,845))
                    vlos = (-0.0175*pha_rectangle)/(4* 3.14159*(2.5/1440)) #LOS Speed
                    flow = vlos/(np.cos(np.radians(alpha)))
                    flow = flow[610:780,80:340] #cut image directly to glacier front
                    ice0.append(flow[36,133]) #glacier + lime squar
                    ice0a.append(flow[53,130]) #tooth triangle lime
                    ice1.append(flow[46,151]) #glacier + black square
                    ice1a.append(flow[61,145]) #tooth triangle black
                    #Atmospheric Noise
                    rockfav = np.array(vlos[830:850,350:370])
                    unravel = rockfav.ravel()
                    median = st.median(unravel)
                    medians.append(median)
                    q1.append(np.percentile(unravel, 25, interpolation = 'midpoint'))
                    q2.append(np.percentile(unravel, 50, interpolation = 'midpoint'))
                    q3.append(np.percentile(unravel, 75, interpolation = 'midpoint'))
                    IQR = np.percentile(unravel, 75, interpolation = 'midpoint') - np.percentile(unravel, 25, interpolation = 'midpoint')
                    iqr.append(IQR)
                    #dates
                    t_start.append(dt.datetime.strptime(name[:15], '%Y%m%d_%H%M%S'))
                    t_end.append(dt.datetime.strptime(name[17:32], '%Y%m%d_%H%M%S')) 

#%%
ice0=[]
ice1=[]
ice0a=[]
ice1a=[]
flist= []
d_list = []
t_start = []
t_end = []
t_diff = []
medians = []
q1 =[]
q2 =[]
q3 =[]
iqr=[]
name_list=[]
slopes=[]

# The lines leading up to the alpha are creation of angles to be used for 
# correcting LOS speed to true flow direction

yr = 1
xr = 1
ys = np.arange(0,674)
xs = np.arange(0,1495)
for y in ys:
    for x in xs:
#        print((y-yr)/(x-xr))
        xy = math.atan((x-xr)/(y-yr))
        xy = math.degrees(xy)
        slopes.append(np.abs(xy))
slopes = np.reshape(slopes, (674,1495))
gamma = 180 - np.array(slopes)
beta = 225 #degrees from 0 north in adf.unw
alpha = 360 - beta - gamma
alpha = np.reshape(alpha, (674,1495))

#Looping through directories to get speeds on 28th and 29th for 
#2nd and 3rd calving events and melange development
#not using rect files since d0729 does not have rect files
for root, dirs, files in os.walk('/data/stor/basic_data/tri_data/rink/proc_data/'):
    for name in sorted(files):
        if name.endswith(('adf.unw')):
            file_paths= os.path.join(root, name)
            flist.append(file_paths) 
#            flist.sort()   
            if file_paths.startswith(('/data/stor/basic_data/tri_data/rink/proc_data/d0728/INT/')):
                with open(file_paths,'rb') as f:
                    temp= f.read()
                    pha_= np.fromfile(file_paths, dtype='>f')
                    pha_[pha_==0] = np.nan
                    pha_rectangle = np.reshape(pha_, (674,1495))
                    vlos = (-0.0175*pha_rectangle)/(4* 3.14159*(2.5/1440)) #LOS Speed
                    flow = vlos/(np.cos(np.deg2rad(alpha)))
                    flow = flow[152:260,380:501] #cut image directly to glacier front
                    ice0.append(flow[18,62]) #glacier + lime
                    ice0a.append(flow[43,39]) #tooth triangle lime
                    ice1.append(flow[44,77]) #glacier + black
                    ice1a.append(flow[60,57]) #tooth triangle black
#                    ice0.append(flow[51,82]) #glacier + red
#                    ice0a.append(flow[55,68]) #tooth triangle darkred
#                    ice1.append(flow[42,76]) #glacier + limegreen
#                    ice1a.append(flow[46,64]) #tooth triangle darkgreen
                    #Atmospheric Noise
                    rockfav = np.array(vlos[310:330,610:620])
                    unravel = rockfav.ravel()
                    median = st.median(unravel)
                    medians.append(median)
                    q1.append(np.percentile(unravel, 25, interpolation = 'midpoint'))
                    q2.append(np.percentile(unravel, 50, interpolation = 'midpoint'))
                    q3.append(np.percentile(unravel, 75, interpolation = 'midpoint'))
                    IQR = np.percentile(unravel, 75, interpolation = 'midpoint') - np.percentile(unravel, 25, interpolation = 'midpoint')
                    iqr.append(IQR)
                    #dates
                    t_start.append(dt.datetime.strptime(name[:15], '%Y%m%d_%H%M%S'))
                    t_end.append(dt.datetime.strptime(name[17:32], '%Y%m%d_%H%M%S'))    
            if file_paths.startswith(('/data/stor/basic_data/tri_data/rink/proc_data/d0729/INT/')):
                with open(file_paths,'rb') as f:
                    temp= f.read()
                    pha_= np.fromfile(file_paths, dtype='>f')
                    pha_[pha_==0] = np.nan
                    pha_rectangle = np.reshape(pha_, (674,1495))
                    vlos = (-0.0175*pha_rectangle)/(4* 3.14159*(2.5/1440))
                    flow = vlos/(np.cos(np.deg2rad(alpha)))
                    flow = flow[152:260,380:501] #[633:700,177:280]
                    ice0.append(flow[18,62]) #glacier + lime
                    ice0a.append(flow[43,39]) #tooth triangle lime
                    ice1.append(flow[44,77]) #glacier + black
                    ice1a.append(flow[60,57]) #tooth triangle black
                    rockfav = np.array(vlos[310:330,610:620])
                    unravel = rockfav.ravel()
                    median = st.median(unravel)
                    medians.append(median)
                    q1.append(np.percentile(unravel, 25, interpolation = 'midpoint'))
                    q2.append(np.percentile(unravel, 50, interpolation = 'midpoint'))
                    q3.append(np.percentile(unravel, 75, interpolation = 'midpoint'))
                    IQR = np.percentile(unravel, 75, interpolation = 'midpoint') - np.percentile(unravel, 25, interpolation = 'midpoint')
                    iqr.append(IQR)
                    #dates
                    t_start.append(dt.datetime.strptime(name[:15], '%Y%m%d_%H%M%S'))
                    t_end.append(dt.datetime.strptime(name[17:32], '%Y%m%d_%H%M%S'))    
                    
#%% Visualize locations on glacier
                    
plt.figure()  
plt.imshow(flow, aspect ='equal' ,norm=colors.SymLogNorm(linthresh = np.amin(5),linscale=0.090, vmin=7.5, vmax=8.5))  
plt.grid()
plt.axis('equal')
plt.colorbar()     
plt.plot([133],[36],'s', color='limegreen', markersize = 8)   
plt.plot([130],[53],'v', color='limegreen', markersize = 8)    
plt.plot([151],[46],'s', color='black', markersize = 8)    
plt.plot([145],[61],'v', color='black', markersize = 8)    
#%%
plt.figure()  
plt.imshow(flow, aspect ='equal' ,norm=colors.SymLogNorm(linthresh = np.amin(5),linscale=0.090, vmin=7.5, vmax=13.5))  
plt.grid()
plt.axis('equal')
plt.colorbar()     
plt.plot([62],[18],'s', color='limegreen', markersize = 8)   
plt.plot([39],[43],'v', color='limegreen', markersize = 8)    
plt.plot([77],[44],'s', color='black', markersize = 8)    
plt.plot([57],[60],'v', color='black', markersize = 8) 

#%% Visualize plotted speeds 

plt.figure()
plt.plot(t_start, ice0, '.', color = 'red')
plt.plot(t_start,ice0a, '.', color ='darkred')

#plt.figure()
plt.plot(t_start,ice1, '.', color = 'limegreen')
plt.plot(t_start,ice1a, '.', color ='darkgreen')

#%%
ice0_vel_meas = np.array(ice0) #red
ice0a_vel_meas = np.array(ice0a) #darkred
ice1_vel_meas = np.array(ice1)   #limegreen
ice1a_vel_meas = np.array(ice1a)#darkgreen


noise_rand = np.array(iqr)
noise_syst = np.array(medians)

#Dt = 2.5 # minutes
#dt_days = Dt/1440 # days
#t = np.linspace(2, 2+len(gla1_vel_meas)*dt_days, len(gla1_vel_meas))
#t[3251:] += .01 # Add time to the end of the record, to simulate a gap in recording
##x = np.random.normal(loc=noise_syst, scale=noise_rand,
##                     size=[4, 30])
t = t_end
for i in range(len(t_end)):
    t[i] = np.datetime64(t_end[i])
t = np.array(t)

#t = t_start

#plt.plot(t,gla1_vel_meas, '.')
# Correct the measurement data with the systematic noise, as in doing the atmospheric correction
gla1_vel_corr = ice0_vel_meas - noise_syst
gla2_vel_corr = ice0a_vel_meas - noise_syst
gla3_vel_corr = ice1_vel_meas - noise_syst
gla4_vel_corr = ice1a_vel_meas - noise_syst

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


# Add in nans to gappy data smoothed velocity, and gappy smoothed uncertainty,
#    for plotting
gap_thresh = np.timedelta64(25, 'm') #0.01 # days
t_gap, vel_run_gap_g1 = nan_into_gap(t_run, vel_run_gla1, gap_thresh)
t_gap, vel_run_gap_g2 = nan_into_gap(t_run, vel_run_gla2, gap_thresh)
t_gap, vel_run_gap_g3 = nan_into_gap(t_run, vel_run_gla3, gap_thresh)
t_gap, vel_run_gap_g4 = nan_into_gap(t_run, vel_run_gla4, gap_thresh)
t_gap, noise_run_gap = nan_into_gap(t_run, noise_run, gap_thresh)   

#%% Calcuate displacements as running sum

time_between_samples = 2.5/1440    
displacement0 = np.array([sum(vel_run_gap_g1[:i]) for i in range(len(vel_run_gap_g1))])*time_between_samples  #green open
displacement0a = np.array([sum(vel_run_gap_g2[:i]) for i in range(len(vel_run_gap_g2))])*time_between_samples  #green closed   

plt.figure()
plt.plot(t_start, displacement0,'--', color = 'limegreen', markersize = 2, label = 'Glacier') #glacier
plt.plot(t_start, displacement0a,'-', color = 'limegreen', markersize = 2,label = 'Tooth')   # tooth
plt.xlabel('Time', fontsize = 16)
plt.ylabel('Displacement (m)', fontsize = 16)
plt.title('Crack', fontweight= 'bold',fontsize = 18)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.axvspan(pd.to_datetime('2014-07-29-05:37:30'),pd.to_datetime('2014-07-30-23:25:00'), alpha= 0.2, color = 'gray', label = 'Mélange')
plt.axvline(pd.to_datetime('2014-07-29-02:52:00'), color='k', linestyle='--', linewidth = 1,label = 'Calving Event')
plt.axvline(pd.to_datetime('2014-07-28-18:02:00'), color='k', linestyle='--', linewidth = 1)  
plt.legend()         
plt.grid()         
   
displacement1 = np.array([sum(vel_run_gap_g3[:i]) for i in range(len(vel_run_gap_g3))])*time_between_samples     
displacement1a = np.array([sum(vel_run_gap_g4[:i]) for i in range(len(vel_run_gap_g4))])*time_between_samples

plt.figure()
plt.plot(t_start, displacement1,'--', color = 'black', markersize = 2, label = 'Glacier')
plt.plot(t_start, displacement1a,'-', color = 'black', markersize = 2, label = 'Tooth')
plt.xlabel('Time', fontsize = 16)
plt.ylabel('Displacement (m)', fontsize = 16)
plt.title('Crack', fontweight= 'bold',fontsize = 18)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.axvspan(pd.to_datetime('2014-07-29-05:37:30'),pd.to_datetime('2014-07-30-23:25:00'), alpha= 0.2, color = 'gray', label = 'Mélange')
plt.axvline(pd.to_datetime('2014-07-29-02:52:00'), color='k', linestyle='--', linewidth = 1,label = 'Calving Event')
plt.axvline(pd.to_datetime('2014-07-28-18:02:00'), color='k', linestyle='--', linewidth = 1)
plt.legend()

#%% Create a first order polynomial to detrend data

t_range = np.arange(0,1702.5, 2.5)

fig = plt.figure()
plt.subplot(2, 1, 1)
plt.title('polyfit, original data set')
plt.plot(t_range, displacement0, 'c', linewidth = 1)

#create line
coeff0 = np.polyfit(t_range, displacement0, 1)
y_poly0 = np.linspace(t_range.min(), t_range.max())
x_poly0 = np.polyval(coeff0, y_poly0)
plt.plot(y_poly0, x_poly0, 'r-', linewidth=2)
plt.subplot(2, 1, 2)
plt.title('detrended')
# we need the original x values here, so we can remove the trend from all points
trend = np.polyval(coeff0, t_range)
# note that simply subtracting the trend might not be enough for other data sets
plt.plot(t_start, displacement0 - trend, '.', color = 'red', linewidth = 2)
plt.plot(t_start, displacement0a - trend, '.', color = 'darkred', linewidth = 2)
fig.show()
plt.ylim(-1.0,1.0,0.5)
plt.title('Tooth Displacement')
plt.ylabel('Detrended Displacement (m)')
plt.xlabel('Time')
plt.axvspan(pd.to_datetime('2014-07-29-05:37:30'),pd.to_datetime('2014-07-30-23:25:00'), alpha= 0.2, color = 'gray', label = 'Mélange')
plt.axvline(pd.to_datetime('2014-07-29-02:52:00'), color='k', linestyle='--', linewidth = 1,label = 'Calving Event')
plt.axvline(pd.to_datetime('2014-07-28-18:02:00'), color='k', linestyle='--', linewidth = 1)
plt.legend()

#%% Detrend for both glacier positions
fig = plt.figure()

plt.title('polyfit, original data set')
plt.plot(t_range, displacement1, 'c', linewidth = 1)

coeff0 = np.polyfit(t_range, displacement0, 1)
y_poly0 = np.linspace(t_range.min(), t_range.max())
x_poly0 = np.polyval(coeff0, y_poly0)

coeff1 = np.polyfit(t_range, displacement1, 1)
y_poly1 = np.linspace(displacement1.min(), displacement1.max())
x_poly1 = np.polyval(coeff1, y_poly1)
#plt.plot(y_poly1, x_poly1, 'r-', linewidth=2)

plt.subplot(2, 1, 1 )
#plt.title('detrended')
# we need the original x values here, so we can remove the trend from all points
trend = np.polyval(coeff0, t_range)
# note that simply subtracting the trend might not be enough for other data sets
plt.plot(t_start, displacement0 - trend, '--', color = 'limegreen', linewidth = 2, label = 'Glacier')
plt.plot(t_start, displacement0a - trend, '-', color = 'limegreen', linewidth = 2, label = 'Ice Tooth')
fig.show()

#plt.title('Tooth Displacement')
plt.ylabel('Detrended Displacement (m)')
plt.xlabel('Date')
plt.axvspan(pd.to_datetime('2014-07-29-04:00:00'),pd.to_datetime('2014-07-30-23:25:00'), alpha= 0.2, color = 'gray')
plt.axvline(pd.to_datetime('2014-07-29-02:52:00'), color='k', linestyle='--', linewidth = 1)
plt.axvline(pd.to_datetime('2014-07-28-18:02:00'), color='k', linestyle='--', linewidth = 1)
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
# we need the original x values here, so we can remove the trend from all points
trend1 = np.polyval(coeff1, t_range)
# note that simply subtracting the trend might not be enough for other data sets
plt.plot(t_start, displacement1 - trend1, '--', color = 'black', linewidth = 2, label = 'Glacier')
plt.plot(t_start, displacement1a - trend1, '-', color = 'black', linewidth = 2, label = 'Ice Tooth')
plt.ylabel('Detrended Displacement (m)')
plt.xlabel('Date')
plt.axvspan(pd.to_datetime('2014-07-29-04:00:00'),pd.to_datetime('2014-07-30-23:25:00'), alpha= 0.2, color = 'gray', label = 'Mélange')
plt.axvline(pd.to_datetime('2014-07-29-02:52:00'), color='k', linestyle='--', linewidth = 1,label = 'Calving Event')
plt.axvline(pd.to_datetime('2014-07-28-18:02:00'), color='k', linestyle='--', linewidth = 1)
plt.legend()
plt.grid()
plt.gcf().autofmt_xdate()
fig.show()

#%%
ice0_vel_meas = np.array(ice0) #red
ice0a_vel_meas = np.array(ice0a) #darkred
ice1_vel_meas = np.array(ice1)   #limegreen
ice1a_vel_meas = np.array(ice1a)#darkgreen


noise_rand = np.array(iqr)
noise_syst = np.array(medians)

#Dt = 2.5 # minutes
#dt_days = Dt/1440 # days
#t = np.linspace(2, 2+len(gla1_vel_meas)*dt_days, len(gla1_vel_meas))
#t[3251:] += .01 # Add time to the end of the record, to simulate a gap in recording
##x = np.random.normal(loc=noise_syst, scale=noise_rand,
##                     size=[4, 30])
t = t_end
for i in range(len(t_end)):
    t[i] = np.datetime64(t_end[i])
t = np.array(t)

#t = t_start

#plt.plot(t,gla1_vel_meas, '.')
# Correct the measurement data with the systematic noise, as in doing the atmospheric correction
gla1_vel_corr = ice0_vel_meas - noise_syst
gla2_vel_corr = ice0a_vel_meas - noise_syst
gla3_vel_corr = ice1_vel_meas - noise_syst
gla4_vel_corr = ice1a_vel_meas - noise_syst

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
t_span = np.timedelta64(120, 'm') #2/24 # days
t_run, vel_run_gla1, noise_run = running_ave(t, t_span, gla1_vel_corr, noise_rand)
t_run, vel_run_gla2, noise_run = running_ave(t, t_span, gla2_vel_corr, noise_rand)
t_run, vel_run_gla3, noise_run = running_ave(t, t_span, gla3_vel_corr, noise_rand)
t_run, vel_run_gla4, noise_run = running_ave(t, t_span, gla4_vel_corr, noise_rand)


# Add in nans to gappy data smoothed velocity, and gappy smoothed uncertainty,
#    for plotting
gap_thresh = np.timedelta64(25, 'm') #0.01 # days
t_gap, vel_run_gap_g1 = nan_into_gap(t_run, vel_run_gla1, gap_thresh)
t_gap, vel_run_gap_g2 = nan_into_gap(t_run, vel_run_gla2, gap_thresh)
t_gap, vel_run_gap_g3 = nan_into_gap(t_run, vel_run_gla3, gap_thresh)
t_gap, vel_run_gap_g4 = nan_into_gap(t_run, vel_run_gla4, gap_thresh)
t_gap, noise_run_gap = nan_into_gap(t_run, noise_run, gap_thresh)   


# Plot the temporally-smoothed data Glacier ( in one graph)
plt.figure()

plt.fill_between(t_gap, vel_run_gap_g1-noise_run_gap, vel_run_gap_g1+noise_run_gap, 
                   color='limegreen',  alpha='.2')
#plt.scatter(t, gla1_vel_corr, s=1, color='limegreen')
plt.plot(t_gap, vel_run_gap_g1, '--', color='limegreen', markersize = 0.5, label = 'glacier')

plt.fill_between(t_gap, vel_run_gap_g2-noise_run_gap, vel_run_gap_g2+noise_run_gap, 
                   color='limegreen', alpha='.2')
#plt.scatter(t, gla2_vel_corr, s=1, color='darkred')
plt.plot(t_gap, vel_run_gap_g2, color='limegreen', markersize =0.5,label = 'tooth')

plt.fill_between(t_gap, vel_run_gap_g3-noise_run_gap, vel_run_gap_g3+noise_run_gap, 
                   color='black', alpha='.2')
#plt.scatter(t, gla3_vel_corr, s=1, color='dimgray')
plt.plot(t_gap, vel_run_gap_g3, '--', color='black', markersize =0.5,label = 'glacier')

plt.fill_between(t_gap, vel_run_gap_g4-noise_run_gap, vel_run_gap_g4+noise_run_gap, 
                   color='black', alpha='.2')
#plt.scatter(t, gla4_vel_corr, s=1, color='black')
plt.plot(t_gap, vel_run_gap_g4, color='black', markersize =0.5, label = 'tooth')



#plt.legend()
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Speeds, with uncertainties (m/d)', fontsize = 18)
plt.title('Velocities (m/d)', fontsize = 24)
plt.axvspan(pd.to_datetime('2014-07-29-05:37:30'),pd.to_datetime('2014-07-30-23:25:00'), alpha= 0.2, color = 'gray', label = 'Mélange')
plt.axvline(pd.to_datetime('2014-07-29-02:52:00'), color='k', linestyle='--', linewidth = 2, label = 'Calving Event')
plt.axvline(pd.to_datetime('2014-07-28-18:02:00'), color='k', linestyle='--', linewidth = 2)

plt.legend(fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)   
plt.grid()

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
                
plt.plot([-230.615],[-1980.85],'v', color='limegreen', markersize = 6)
plt.plot([-230.759],[-1980.91],'v', color='darkgreen', markersize = 6)                                  
plt.plot([-230.601],[-1981.09],'s', color='red', markersize = 6)
plt.plot([-230.709],[-1981.12],'s', color='darkred', markersize = 6)     

#%% SUBPLOTS

fig, axes = plt.subplots(2, 2)
#plt.figure()
#axes = plt.subplot2grid((2,2),(1,0), colspan = 1)
#axes0 = plt.subplot2grid((2,2),(1,1), colspan = 1)
##axes0 = plt.subplot2grid((3,3),(2,0), colspan = 2)
#axes1 = plt.subplot2grid((2,2), (0,0), colspan=1)
#axes2 = plt.subplot2grid((2,2), (0,1), colspan=1)

    
axes[1,0].fill_between(t_gap, vel_run_gap_g1-noise_run_gap, vel_run_gap_g1+noise_run_gap, 
                   color='limegreen', alpha='.2')
#plt.scatter(t, gla1_vel_corr, s=1, color='red')
axes[1,0].plot(t_gap, vel_run_gap_g1, '--', color='limegreen', markersize = 0.5)

axes[1,0].fill_between(t_gap, vel_run_gap_g2-noise_run_gap, vel_run_gap_g2+noise_run_gap, 
                   color='limegreen', alpha='.2')
#plt.scatter(t, gla2_vel_corr, s=1, color='darkred')
axes[1,0].plot(t_gap, vel_run_gap_g2, color='limegreen', markersize =0.5)

axes[1,0].fill_between(t_gap, vel_run_gap_g3-noise_run_gap, vel_run_gap_g3+noise_run_gap, 
                   color='black', alpha='.2')
#plt.scatter(t, gla3_vel_corr, s=1, color='dimgray')
axes[1,0].plot(t_gap, vel_run_gap_g3, '--', color='black', markersize =0.5)

axes[1,0].fill_between(t_gap, vel_run_gap_g4-noise_run_gap, vel_run_gap_g4+noise_run_gap, 
                   color='black', alpha='.2')
#plt.scatter(t, gla4_vel_corr, s=1, color='black')
axes[1,0].plot(t_gap, vel_run_gap_g4, color='black', markersize =0.5)


#plt.legend()
axes[1,0].set_xlabel('Date', fontsize = 13)
axes[1,0].set_ylabel('Speeds (m/d)', fontsize = 13)
#axes.set_title('Velocities (m/d)', fontsize = 24)
#plt.axvspan(pd.to_datetime('2014-07-25-12:45:00'),pd.to_datetime('2014-07-25-15:15:00'), alpha= 0.4, color = 'darkgray', label = 'TRI motion error')
axes[1,0].axvspan(pd.to_datetime('2014-07-29-04:00:00'),pd.to_datetime('2014-07-30-23:25:00'), alpha= 0.2, color = 'gray', label = 'Mélange')
#plt.axvline(pd.to_datetime('2014-07-26-10:00:00'), color='k', linestyle=':', linewidth = 3, label = 'Calving Event')
axes[1,0].axvline(pd.to_datetime('2014-07-29-02:52:00'), color='k', linestyle='--', linewidth = 1, label = 'Calving Event')
axes[1,0].axvline(pd.to_datetime('2014-07-28-18:02:00'), color='k', linestyle='--', linewidth = 1)
axes[1,0].legend()  
axes[1,0].grid()

axes[1,1].pcolormesh(e_eas/1000, n_nor/1000, mli_rot, vmin=0, vmax=10, cmap = 'gray')#20)
axes[1,1].axis('equal')
axes[1,1].grid(True)
axes[1,1].set_xlabel('Easting (km)', fontsize = 13)
axes[1,1].set_ylabel('Northing (km)', fontsize = 13)                                               
axes[1,1].plot([-230.601],[-1981.09],'s', color='black', markersize = 6)
axes[1,1].plot([-230.794],[-1981.12],'v', color='black', markersize = 6) 
axes[1,1].plot([-230.616],[-1980.9],'s', color='limegreen', markersize = 6)
axes[1,1].plot([-230.816],[-1980.93],'v', color='limegreen', markersize = 6) 

axes[0,0].plot(t_start, displacement0 - trend, '--', color = 'limegreen', markersize = 1, label = 'Glacier')
axes[0,0].plot(t_start, displacement0a - trend, '-', color = 'limegreen', markersize = 1, label = 'Ice Tooth')
axes[0,0].set_ylim(-0.25,1.7,0.5)
#axes2.set_title('Tooth Displacement')
axes[0,0].set_ylabel('Detrended Displacement (m)', fontsize = 13)
#axes[0,0].set_xlabel('Time', fontsize = 13)
axes[0,0].axvspan(pd.to_datetime('2014-07-29-04:00:00'),pd.to_datetime('2014-07-30-23:25:00'), alpha= 0.2, color = 'gray', label = 'Mélange')
axes[0,0].axvline(pd.to_datetime('2014-07-29-02:52:00'), color='k', linestyle='--', linewidth = 1,label = 'Calving Event')
axes[0,0].axvline(pd.to_datetime('2014-07-28-18:02:00'), color='k', linestyle='--', linewidth = 1)
axes[0,0].legend()
axes[0,0].grid()
    

axes[0,1].plot(t_start, displacement1 - trend1, '--', color = 'black', markersize = 1, label = 'Glacier')
axes[0,1].plot(t_start, displacement1a - trend1, '-', color = 'black', markersize = 1, label = 'Ice Tooth')
axes[0,1].set_ylim(-0.25,1.7,5.5)
#plt.title('Tooth Displacement')
#axes[0,1].set_ylabel('Detrended Displacement (m)', fontsize = 13)
#axes[0,1].set_xlabel('Time', fontsize = 13)
axes[0,1].axvspan(pd.to_datetime('2014-07-29-04:00:00'),pd.to_datetime('2014-07-30-23:25:00'), alpha= 0.2, color = 'gray', label = 'Mélange')
axes[0,1].axvline(pd.to_datetime('2014-07-29-02:52:00'), color='k', linestyle='--', linewidth = 1,label = 'Calving Event')
axes[0,1].axvline(pd.to_datetime('2014-07-28-18:02:00'), color='k', linestyle='--', linewidth = 1)
axes[0,1].legend()
axes[0,1].grid()
  

plt.gcf().autofmt_xdate()


fig.align_ylabels(axes[:])    