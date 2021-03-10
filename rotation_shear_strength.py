#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 18:28:14 2020

@author: eswaninger
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:28:09 2020

@author: eswaninger
"""


#%% Modules

import matplotlib.pyplot as plt
import numpy as np
import Cython as cy
import setuptools as sts
import cftime as cf
from netCDF4 import Dataset
import netCDF4 as nc
import os
import math
import pandas as pd
#%% Data
os.chdir('/data/stor/basic_data/spatial_data/greenland/bedmap/')
filename = 'BedMachineGreenland-2017-09-20.nc'

file = nc.Dataset(filename)

ice_thickness = file['thickness'] #meters
bed_elevation = file['bed'] #meters
mask = file['mask'] #0 = ocean, 1 = ice-free land, 2 = grounded ice, 3 = floating ice, 4 = non-Greenland land
sur = file['surface']

#%%
plt.figure()
plt.imshow(ice_thickness[8960:9020,2780:2840]) 
plt.title('ice thickness')
#plt.plot([62.54],[115.2],'o', color='red', markersize = 6)  

plt.figure()
plt.imshow(sur[8960:9020,2780:2840], vmin = 0, vmax = 100)
plt.title('surface elevation')


plt.plot([62.54],[115.2],'o', color='red', markersize = 6)  

#%% Array of ice thicknesses at terminus and corresponding surface elevations
iceth = np.array(ice_thickness[8960:9020,2780:2840])
#ice_block = iceth[27:33,33:36]
unice = ice_block.ravel() #list of ice thicknesses
surth = np.array(sur[8960:9020,2780:2840])
#sur_block = surth[27:33,33:36]
unsur = sur_block.ravel() #list of surface elevations
D= np.array(bed_elevation[8960:9020,2780:2840])
#D = D[27:33,33:36]
unD = D.ravel() #list of waterdepths 

floating= surth/iceth #if 0.10 ice is at neutral buoyancy, if below then its below buoyancy, if above than greater than neutral buoyancy
plt.imshow(floating, vmin = 0, vmax = 0.2)
plt.colorbar()
plt.title('nearness to neutral buoyancy (0.1)')

#%%
geoid_H = 31.657 #m
#%% Force Variables 1 near shadow (high end value)
L= 1340 #m Length
W_s= 230 #m width at shadow
W_t=  340 #m width at tooth
#H = 520 #m height
H = np.arange(450, 700, 10)
#H_bedmach = unice
pw = 1025   #sea water density
pi = 920    #ice density
g = 9.81    #gravitational acceleration
a = 427     #TRI elevation
b = 63 - geoid_H      #upglacier elevation
c = 54 - geoid_H  #arcticdem terminus elevation
#c_bedmach = unsur #surface elevation list BedMachine
ab = 5700 #distance between tri and glacier 
bc = 75 #horizontal distance between c and b

alpha = math.atan((a-b)/ab)
alpha = np.rad2deg(alpha)
    
beta= math.atan((b-c)/bc)
beta = np.rad2deg(beta)
delta = alpha + beta 

B = math.sin(np.deg2rad(delta))*bc #elevation between hinge and top of radar graze angle
y = 180 - 90- delta

zeta = np.rad2deg(math.acos(B/W_s))
theta = 180 - y - zeta
psi = theta - beta

#C1 = []
#for i in theta:
#    cc = math.tan(np.deg2rad(i))*(W)
#    C1.append(cc)
#C1 = np.array(C1)
#
#C2 =[]
#for i in psi:
#    cc = math.tan(np.deg2rad(i))*(W)
#    C2.append(cc)
#C2 = np.array(C2)

        
C1 = math.tan(np.deg2rad(theta))*(W_s)
C2 = math.tan(np.deg2rad(psi))*(W_s)
C3 = C1-C2
#C3 = c/(math.cos(np.deg2rad(psi)))
#C1 = C2+C3

A= H*W_t     
A1 = (1/2)*C2*W_s
A2 = (C3)*W_s
Vi = L*W_s*H #volume of ice with different heights
Va = A1*L + A2*L
Vs = Vi - Va #volume of submerged ice with different heights
    
#Vs=[]
#for i in Va:
#    vv = Vi[15] - i 
#    Vs.append(vv)
#Vs= np.array(Vs)

Fg = -pi*g*Vi
Fb = pw*g*Vs
Mg = Fg*200
Mb = Fb*200
Fc = - (Fb + Fg)

P = (Fc/A[1])/1000



#
#%% Plot Yield Strengths with various iceberg heights
plt.figure()
plt.plot(H, P, color = 'r', label ='θ = 14°')
plt.ylabel(' Shear Stress (kPa)',  fontsize = 13)
plt.xlabel('Terminus thickness (m)', fontsize = 13)
#plt.axvspan(400,500, color='k', linewidth = 1, alpha = 0.2)  
plt.legend(loc= 3, prop={'size': 11})
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid()
#%% Shear Strength with theta array 0 - 11 degrees
plt.figure()
plt.plot(theta, P0, color = 'r')
plt.ylabel(' Shear Strength (kPa)',  fontsize = 13)
plt.xlabel('Theta (θ)', fontsize = 13)
#plt.axvline(530000, color='k', linestyle='--', linewidth = 2)  
#plt.axhline(11, color='k', linestyle='--', linewidth = 2)  

plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid()

#%%
fig, ax1 = plt.subplots()
ax1.plot(P,theta_array,  color = 'r')
ax1.set_xlabel(' Shear Stress (Pa)',  fontsize = 13)
ax1.set_ylabel('Theta (θ)', fontsize = 13)
#ax1.axvline(11, color='k', linestyle='--', linewidth = 2)  
#ax1.axhline(280000, color='k', linestyle='--', linewidth = 2)  
ax1.tick_params(axis='both', which='major', labelsize=12)

ax2 = ax1.twinx()
ax2.plot(P, H, color = 'g')
ax2.set_ylabel(' Height',  fontsize = 13)
ax2.tick_params(axis='both', which='major', labelsize=12)

ax1.grid()

#%% Plot Shear strength bw tooth and 3rd iceberg
plt.figure()
plt.plot(H, Pcf[0], label ='μ = 0.40')
plt.plot(H, Pcf[1], label ='μ = 0.50')
plt.plot(H, Pcf[2], label ='μ = 0.65')
plt.plot(H, Pcf[3], label ='μ = 0.80')
plt.plot(H, Pcf[4], label ='μ = 1.00')
plt.plot(H, Sxx , '--', color ='k', label = 'Maximum Shear Stress' )
plt.xlabel('Ice thickness (m)', fontsize = 13)
plt.ylabel('Shear Strength of Terminus Ice (Pa)', fontsize = 13)
plt.tick_params(axis='x', which='major', labelsize=11)
plt.tick_params(axis='y', which='major', labelsize=11)
plt.legend()
plt.grid()

#plt.ylabel("Shear Strength (Pa)")
#plt.xlabel('area (m**2)')

#%%
plt.figure()
plt.plot(C2,Cs)

fig, axes = plt.subplots(nrows= 4, clear = True)
y = np.arange(0,60, 1)
axes[0].plot(theta, P, '.', color='blue', markersize = 3.5)
#axes[0,0].plot(np.unique(area), np.poly1d(np.polyfit(area, P, 2))(np.unique(P)))
axes[0].set_ylabel('Shear Strength (kPa)', fontsize = 12)
axes[0].yaxis.set_label_coords(-0.09,0.5)
axes[0].tick_params(axis='x', which='major', labelsize=12)
axes[0].tick_params(axis='y', which='major', labelsize=12)
axes[0].grid() 


#axes[1].plot(theta,Sxx,'.', color='blue',markersize = 3.5)
#axes[1].set_ylabel('Sxx (kPa)', fontsize = 12)
#axes[1].yaxis.set_label_coords(-0.09,0.5)
#axes[1].tick_params(axis='x', which='major', labelsize=12)
#axes[1].tick_params(axis='y', which='major', labelsize=12)
#axes[1].grid() 
#axes[1].axvspan(0.75,0.85, alpha= 0.2, color = 'gray')

axes[2].plot(theta,Fnet, '.',color='blue', markersize =3.5)
axes[2].set_ylabel('Net-Force (N)', fontsize = 12)
axes[2].yaxis.set_label_coords(-0.09,0.5)
axes[2].tick_params(axis='x', which='major', labelsize=12)
axes[2].tick_params(axis='y', which='major', labelsize=12)
axes[2].grid() 


axes[3].plot(theta,T, '.',color='blue', markersize =3.5)
axes[3].set_ylabel('Torque Moment (Nm)', fontsize = 12)
axes[3].yaxis.set_label_coords(-0.09,0.5)
axes[3].tick_params(axis='x', which='major', labelsize=12)
axes[3].tick_params(axis='y', which='major', labelsize=12)
axes[3].grid() 
axes[3].axis('auto')
axes[3].set_xlabel("Aspect Ratio", fontsize=12)


plt.gcf().autofmt_xdate()
