#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jan 30 16:42:41 2020

@author: eswaninger

Environment: imgenv_yml
"""
#%%
''' This script is for observation of Atmospheric Heterogenity at closest
fjord wall
'''
#%% Modules

import numpy as np
import matplotlib.pyplot as plt
import os 
import datetime as dt
import statistics as st
import pandas as pd
from scipy.stats import iqr
import matplotlib.cbook as cbook
#from scipy.spatial import Delaunay
import matplotlib.patches as patches
from PIL import Image

#%%
rock0 =[]
gla =[]
#
flist= []
d_list = []
t_start = []
t_end = []
t_diff = []
medians = []
means = []
t_mean = []
stds = []
systs = []
q1 =[]
q2 =[]
q3 =[]
iqrs=[]
inter =[]
stats={}
iqrstats=[]
snratios=[]
name_list=[]

for root, dirs, files in os.walk('/data/stor/basic_data/tri_data/rink/proc_data/d0728/INT/REC/'):
    for name in sorted(files):
        if name.endswith(('adf.unw.rec')):
            file_paths= os.path.join(root, name)
            flist.append(file_paths) 
#            flist.sort()   
            t_start.append(dt.datetime.strptime(name[:15], '%Y%m%d_%H%M%S'))
            t_end.append(dt.datetime.strptime(name[17:32], '%Y%m%d_%H%M%S'))                                          
            if file_paths.startswith(('/data/stor/basic_data/tri_data/rink/proc_data/d0728/INT/REC/')):
                with open(file_paths,'rb') as f:
                    temp= f.read()
                    pha_= np.fromfile(file_paths, dtype='>f')
                    pha_[pha_==0] = np.nan
                    pha_rectangle = np.reshape(pha_, (1559,845))
                    vlos = (-0.0175*pha_rectangle)/(4* 3.14159*(2.5/1440)) #LOS Speeds
                    gla.append(vlos[673,233])
                    rock0.append(vlos[665,355]) # one rock velocity
                    #Atmosphere
                    rockfav = np.array(vlos[600:620,360:380]) #rock square for noise analysis
                    unravel = rockfav.ravel()
                    median = st.median(unravel)               #medians of rock wall noise
                    medians.append(median)
                    syst = unravel-median                       # rock walls without median values
                    systs.append(np.std(syst))                          # left over rock wall LOS velocities
                    means.append(np.mean(unravel))
                    true_mean = np.mean(unravel) +2*(np.std(unravel)/np.sqrt(400))
                    t_mean.append(true_mean)
                    std = np.std(unravel)
                    stds.append(std)
                    snratio= unravel/np.std(unravel)
                    snratios.append(snratio)
                    q1.append(np.percentile(unravel, 25, interpolation = 'midpoint'))
                    q2.append(np.percentile(unravel, 50, interpolation = 'midpoint'))
                    q3.append(np.percentile(unravel, 75, interpolation = 'midpoint'))
                    IQR = np.percentile(unravel, 75, interpolation = 'midpoint') - np.percentile(unravel, 25, interpolation = 'midpoint')
                    iqrs.append(IQR)
                    inter_quart = iqr(unravel)
                    inter.append(inter_quart)
             

#%%   Dictionary of IQR stats
stats = {}
# Compute the boxplot stats (as in the default matplotlib implementation)
stats['A'] = cbook.boxplot_stats(unravel, labels='A')[0]
stats['B'] = cbook.boxplot_stats(unravel, labels='B')[0]
stats['C'] = cbook.boxplot_stats(unravel, labels='C')[0]

# For box A compute the 1st and 99th percentiles
stats['A']['q1'], stats['A']['q3'] = np.percentile(unravel, [1, 99])
# For box B compute the 10th and 90th percentiles
stats['B']['q1'], stats['B']['q3'] = np.percentile(unravel, [10, 90])
# For box C compute the 25th and 75th percentiles (matplotlib default)
stats['C']['q1'], stats['C']['q3'] = np.percentile(unravel, [25, 75])

fig, ax = plt.subplots(1, 1)
# Plot boxplots from our computed statistics
ax.bxp([stats['A'], stats['B'], stats['C']], positions=range(3))
#%%

# Create figure and axes
fig, axes = plt.subplots()

# Display the image
axes.imshow(vlos)

# Create a Rectangle patch
rect = patches.Rectangle((360, 620), 20, 20, linewidth=1, edgecolor='orange', facecolor='none')
plt.plot(328, 832, '.', color = 'r')
plt.plot(233, 673, '.', color = "b")


# Add the patch to the Axes
axes.add_patch(rect)

plt.show()
                    #%%
noise_rand = np.array(medians)
noise_syst = np.array(stds)

#Dt = 2.5 # minutes
#dt_days = Dt/1440 # days
#t = np.linspace(2, 2+len(gla1_vel_meas)*dt_days, len(gla1_vel_meas))
#t[3251:] += .01 # Add time to the end of the record, to simulate a gap in recording
##x = np.random.normal(loc=noise_syst, scale=noise_rand,
##                     size=[4, 30])
t = t_start
for i in range(len(t_end)):
    t[i] = np.datetime64(t_end[i])
t = np.array(t)



# Correct the measurement data with the systematic noise, as in doing the atmospheric correction
vel_corr = gla - noise_rand
# There shouldn't be a need, I don't think, to modify these function
#%% Calculate Running Averages, add nans to discontinuous data, and plot

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
t_span = np.timedelta64(60, 'm')
t_run, vel_run, noise_run = running_ave(t, t_span, vel_corr, noise_rand)


# Add in nans to gappy data smoothed velocity, and gappy smoothed uncertainty,
#    for plotting
gap_thresh = np.timedelta64(20, 'm')# days
t_gap, vel_run_gap = nan_into_gap(t, vel_run, gap_thresh)
t_gap, noise_run_gap = nan_into_gap(t, noise_run, gap_thresh)


#%%  Plot data, with corrections for error
fig, ax = plt.subplots(nrows=4, num=1, sharex=True, clear=True)
ax[0].scatter(t, rock0, marker ='.', color = 'r')
#ax[0].scatter(t, vel_corr, marker ='.', label='Corrected for noise')
ax[0].set_title('Raw Rock Wall Velocity \n(m/d)')
#ax[0].legend()
#ax[0].axis('equal')
#ax[0].get_xaxis().set_visible(False)
ax[0].grid(axis='both')

ax[1].scatter(t, noise_rand, marker = '.', color = 'orange')
ax[1].set_title('Raw Rock Wall Medians (m/d)', fontsize=12) #'Random \nerror (m/d)')
#ax[1].get_xaxis().set_visible(False)
ax[1].grid(axis = 'both')

#ax[2].scatter(t, noise_syst,marker ='.')
#ax[2].set_title('Corrected Rock Wall (Vel - Median)') 
#ax[2].get_xaxis().set_visible(False)

ax[2].scatter(t, stds,marker = '.', color = 'green')
ax[2].set_title('Rock Wall Standard Deviation (σ)') #Systematic \nerror (m/d)')
#ax[2].get_xaxis().set_visible(False)
ax[2].grid(axis = 'both')

ax[3].fill_between(t, vel_corr-noise_syst, vel_corr+noise_syst, 
                   color='tab:blue', alpha='.5', linewidth = 1.5)
ax[3].plot(t, vel_corr, color='tab:blue', label='Unaveraged velocity', linewidth = .75)
ax[3].plot(t, gla, color = 'r', linewidth = .75)
ax[3].set_title('Glacier Velocities w/ Uncertainty (m/d)')
ax[3].grid(axis = 'both')

ax[3].plot(t_gap, vel_run_gap, color='tab:green', label='Averaged velocity')

ax[3].set_xlabel('Time (d)')
ax[3].grid()


plt.xlabel('Time (d)', fontsize =12)
#plt.axvspan(pd.to_datetime('2014-07-25-12:45:00'),pd.to_datetime('2014-07-25-15:15:00'), alpha= 0.7, color = 'darkgray', label = 'TRI motion error')
#plt.axvspan(pd.to_datetime('2014-07-29-05:37:30'), pd.to_datetime('2014-07-30-23:25:00'), alpha= 0.2, color = 'gray', label = 'Mélange')
#plt.axvline(pd.to_datetime('2014-07-26-10:00:00'), color='k', linestyle=':', linewidth = 3, label = 'Calving Event')
plt.axvline(pd.to_datetime('2014-07-29-02:52:00'), color='k', linestyle='--', linewidth = 2, label = 'Calving Event')
plt.axvline(pd.to_datetime('2014-07-28-18:02:00'), color='k', linestyle='--', linewidth = 2)
ax[3].legend(loc = 'upper right', fontsize = 8)
plt.grid()


#%% DEFINE FUNCTIONS

#%%
# Plot the temporally-smoothed data
fig, ax = plt.subplots(nrows=1, num=2, sharex=True, clear=True)

ax.fill_between(t, vel_corr-noise_rand, vel_corr+noise_rand, 
                   color='tab:orange', alpha='.2')
ax.fill_between(t_gap, vel_run_gap-noise_run_gap, 
                vel_run_gap+noise_run_gap, 
                   color='tab:green', alpha='.2')
ax.plot(t, vel_corr, color='tab:orange', label='Unaveraged velocity')
ax.plot(t_gap, vel_run_gap, color='tab:green', label='Averaged velocity')

ax.set_xlabel('Time (d)')
ax.set_ylabel('Speeds, with uncertainties (m/d)')
ax.legend()
ax.grid()
#ax[0].plot(t, vel_meas, label='Measured')
#ax[0].plot(t, vel_corr, label='Corrected for atmos')
#ax[0].set_ylabel('Speed (m/d)')
#ax[0].legend()
           
#%%
fig, ax = plt.subplots(nrows=3, num=1, sharex=True, clear=True)

ax[0].scatter(t, medians, marker = '.',linewidth =1)
ax[0].set_ylabel('Medians \nof Fjord Wall (m/d)')
ax[0].grid()
ax[0].axvspan(pd.to_datetime('2014-07-29-05:37:30'), pd.to_datetime('2014-07-30-23:25:00'), alpha= 0.2, color = 'gray', label = 'Mélange')
ax[0].axvline(pd.to_datetime('2014-07-29-02:52:00'), color='k', linestyle='--', linewidth = 2, label = 'Calving Event')
ax[0].axvline(pd.to_datetime('2014-07-28-18:02:00'), color='k', linestyle='--', linewidth = 2)

ax[1].scatter(t, iqrs,marker ='.',linewidth =1)
ax[1].set_ylabel('Interquartile Range \nof Fjord Wall (m/d)')
ax[1].grid()
ax[1].axvspan(pd.to_datetime('2014-07-29-05:37:30'), pd.to_datetime('2014-07-30-23:25:00'), alpha= 0.2, color = 'gray', label = 'Mélange')
ax[1].axvline(pd.to_datetime('2014-07-29-02:52:00'), color='k', linestyle='--', linewidth = 2, label = 'Calving Event')
ax[1].axvline(pd.to_datetime('2014-07-28-18:02:00'), color='k', linestyle='--', linewidth = 2)

ax[2].plot(t, gla, linewidth =1.5, label = "Raw", color = 'orange')

#ax[2].set_ylabel('Raw \nGlacier Velocity (m/d)')
#ax[2].legend()
#ax[2].grid()
#ax[2].axvspan(pd.to_datetime('2014-07-29-05:37:30'), pd.to_datetime('2014-07-30-23:25:00'), alpha= 0.2, color = 'gray', label = 'Mélange')
ax[2].axvline(pd.to_datetime('2014-07-29-02:52:00'), color='k', linestyle='--', linewidth = 2, label = 'Calving Event')
ax[2].axvline(pd.to_datetime('2014-07-28-18:02:00'), color='k', linestyle='--', linewidth = 2)
ax[2].set_ylim(-3, 25)

#ax[3].fill_between(t, vel_corr-noise_rand, vel_corr+noise_rand, 
#                   color='tab:orange', alpha='.2')
#ax[3].plot(t, vel_corr, color='tab:orange',linewidth =1)
#ax[3].set_ylabel('Corrected Speed \nw/ Uncertainty (m/d)')


ax[2].fill_between(t_gap, vel_run_gap-noise_run_gap, 
                vel_run_gap+noise_run_gap, 
                   color='tab:green', alpha = '.5')
ax[2].plot(t_gap, vel_run_gap, color='tab:green', label = "Running Average")
ax[2].set_ylabel('Glacier Velocity \n(m/d)')
ax[2].legend()
ax[2].set_ylim(-3, 25)
ax[2].grid()


ax[0].yaxis.set_label_coords(-0.05,0.5)
ax[1].yaxis.set_label_coords(-0.05,0.5)
ax[2].yaxis.set_label_coords(-0.05,0.5)
#ax[3].yaxis.set_label_coords(-0.05,0.5)
plt.setp(ax[0].get_yticklabels(),visible=True)
#fig.align_ylabels(ax[:, 3])

plt.xlabel('Date')
plt.gcf().autofmt_xdate() 

#plt.axvspan(pd.to_datetime('2014-07-25-12:45:00'),pd.to_datetime('2014-07-25-15:15:00'), alpha= 0.4, color = 'darkgray', label = 'TRI motion error')
plt.axvspan(pd.to_datetime('2014-07-29-05:37:30'), pd.to_datetime('2014-07-30-23:25:00'), alpha= 0.2, color = 'gray', label = 'Mélange')
#plt.axvline(pd.to_datetime('2014-07-26-10:00:00'), color='k', linestyle=':', linewidth = 3, label = 'Calving Event')
plt.axvline(pd.to_datetime('2014-07-29-02:52:00'), color='k', linestyle='--', linewidth = 2, label = 'Calving Event')
plt.axvline(pd.to_datetime('2014-07-28-18:02:00'), color='k', linestyle='--', linewidth = 2)