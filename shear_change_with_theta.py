#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 19:25:32 2020

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
import Cython as c
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
plt.imshow(ice_thickness[8875:9050,2750:2925]) #[2813.83,8986.85]
plt.title('ice thickness')
plt.plot([62.54],[115.2],'o', color='red', markersize = 6)  

plt.figure()
plt.imshow(sur[8875:9050, 2750:2925])
plt.title('surface elevation')


plt.plot([62.54],[115.2],'o', color='red', markersize = 6)  

#%% Array of ice thicknesses at terminus and corresponding surface elevations
iceth = np.array(ice_thickness[8875:9050, 2750:2925])
iceth = iceth[111:121,63:69]
unice = iceth.ravel() #list of ice thicknesses
surth = np.array(sur[8875:9050, 2750:2925])
surth = surth[111:121,63:69]
unsur = surth.ravel() #list of surface elevations
D= np.array(bed_elevation[8875:9050, 2750:2925])
D = -D[111:121,63:69]
unD = D.ravel() #list of waterdepths 
#%% Force Variables

L= 1340 #m Length
W=  340 #m width
floating= surth/iceth #if 0.10 ice is floating, if below then its below buoyancy
H= np.array([400,450,500,550,600])   #iceberg height
E = W/H     #aspect ratio
h = np.array([41,46,52,57,62])
D = np.array([974,994,1016,1032,969])
hdiff= (H-h)
pw = 1020   #sea water density
pi = 920    #ice density
g = 9.81    #gravitational acceleration
theta = np.arange(0,8,1.6)
theta = np.deg2rad(theta)   #iceberg rotation


   #Area of idealized iceberg  
A= E*H**2      
 
#A1 = []
#tri_vol=[]
#for i in theta:
#    a = (1/2)*((A)*np.tan(i)) #Area of triangle dipped below surface
#    b = (1/2)*((A)*np.tan(i))*L #volume of triangle
#    A1.append(a)
#    tri_vol.append(b)
#    
#A2 = []
#rect_vol =[]
#for i in theta:
#    a = (A)*((pi/pw)-((1/2)*E*np.tan(i))) #Area of submerged rectangle
#    b = (A)*((pi/pw)-((1/2)*E*np.tan(i)))*L #volume of rectanggle
#    A2.append(a)
#    rect_vol.append(b)

Vt= []
Vr= []
for i in theta:
    a = ((1/2)*((A)*np.tan(8)))
    b = ((A)*((pi/pw)-((1/2)*E*np.tan(i)))) #VOlume submerged
    Vt.append(a)
    Vr.append(b)


#tri_vol=np.array(tri_vol)
#rect_vol= np.array(rect_vol)

V= L*W*H
Vs = np.array(Vs)
#haf= h - H*(1- (pi/pw)) #height above/below floatation
#hab = np.arange(-4, 0, 0.0285)
Fg = -pi*g*V
Fb = pw*g*Vs

Fnet = Fg + Fb #Net force
    
#T  = Fb*((H + haf)/2)*(np.sin(theta)) #torque moment
#
# Shear failure of ice with coefficient of friction
Cf = np.array([0.4,0.65,0.8,1.0])
Co = 0
#Hl = np.array([450, 500, 550, 600])
#Tc = Co + Cf*pi*g*H
#S4 = Ss*0.4
#S6 = Ss*0.65
#S8 = Ss*0.8
#S10 = Ss*1.0
#Yield Strength kPa
#Ss = pi*g*H*(1-(pw/pi)*(D/H)) #only for grounded glaciers
#S4 = Ss*0.4
#S6 = Ss*0.65
#S8 = Ss*0.8
#S10 = Ss*1.0
#max shear stress that needs to be exceeded
Sxx = (1/2)*(pi*g*H)*(1-((pw/pi)*(D/H)**2)) 

#exceed = Ss-Sxx 
#Sxx= Sxx/1000
P = Fnet/A
#P = P/1000
#%% Plot Yield Strengths with various coefficients of friction and iceberg heights
plt.plot(Sxx,S4,  color = 'r', label ='μ = 0.40')
plt.plot(Sxx,S6,color = 'g',label ='μ = 0.65')
plt.plot(Sxx,S8, color = 'b',label ='μ = 0.80')
plt.plot(Sxx,S10, color = 'm', label ='μ = 1.00')
plt.xlabel('Maximum Shear Stress (Pa)')
plt.ylabel('Yield Strength of Ice (Pa)')
plt.legend()
plt.grid()
#%%

plt.plot(Vs, '.')
#%% Plot Shear strength bw tooth and 3rd iceberg
plt.figure()
plt.plot(H, P,  color = 'r')
plt.xlabel('Ice thickness (m)')
plt.ylabel('Shear Strength of Ice Tooth (kPa)')
plt.grid()

#plt.ylabel("Shear Strength (Pa)")
#plt.xlabel('area (m**2)')


#%%

fig, axes = plt.subplots(nrows= 2, ncols=2, clear = True)
y = np.arange(0,60, 1)
axes[0,0].plot(E, P, '.', color='blue', markersize = 3.5)
#axes[0,0].plot(np.unique(area), np.poly1d(np.polyfit(area, P, 2))(np.unique(P)))
axes[0,0].set_ylabel('Shear Strength (kPa)', fontsize = 12)
axes[0,0].yaxis.set_label_coords(-0.09,0.5)
axes[0,0].tick_params(axis='x', which='major', labelsize=12)
axes[0,0].tick_params(axis='y', which='major', labelsize=12)
axes[0,0].grid() 
#axes[0].axvspan(0.75,0.85, alpha= 0.2, color = 'gray')

axes[0,1].plot(E,Sxx,'.', color='blue',markersize = 3.5)
axes[0,1].set_ylabel('Shear Stress (Pa)', fontsize = 12)
axes[0,1].yaxis.set_label_coords(-0.09,0.5)
axes[0,1].tick_params(axis='x', which='major', labelsize=12)
axes[0,1].tick_params(axis='y', which='major', labelsize=12)
axes[0,1].grid() 
axes[0,1].set_xlabel("Aspect Ratio", fontsize=12)
#axes[1].axvspan(0.75,0.85, alpha= 0.2, color = 'gray')

axes[1,0].plot(E,Fnet, '.',color='blue', markersize =3.5)
axes[1,0].set_ylabel('Net-Force (N)', fontsize = 12)
axes[1,0].yaxis.set_label_coords(-0.09,0.5)
axes[1,0].tick_params(axis='x', which='major', labelsize=12)
axes[1,0].tick_params(axis='y', which='major', labelsize=12)
axes[1,0].grid() 
#axes[2].axvspan(0.75,0.85, alpha= 0.2, color = 'gray')

axes[1,1].plot(E,S4, color='red', markersize =3.5,label ='μ = 0.40')
axes[1,1].plot(E,S6, color='blue', markersize =3.5,label ='μ = 0.65')
axes[1,1].plot(E,S8, color='green', markersize =3.5,label ='μ = 0.80')
axes[1,1].plot(E,S10, color='magenta', markersize =3.5,label ='μ = 1.0')
axes[1,1].set_ylabel('Yield Strength (Pa)', fontsize = 12)
axes[1,1].yaxis.set_label_coords(-0.09,0.5)
axes[1,1].tick_params(axis='x', which='major', labelsize=12)
axes[1,1].tick_params(axis='y', which='major', labelsize=12)
axes[1,1].grid() 
axes[1,1].axis('auto')
axes[1,1].set_xlabel("Aspect Ratio", fontsize=12)
#axes[1,1].axvspan(0.75,0.85, alpha= 0.2, color = 'gray')


plt.gcf().autofmt_xdate()
#%% For one idealized geometry

L= 1420 #m
W=  400 #m
H= 500
E = W/H 
h = 50
hdiff= (H-h)
pw = 1020
pi = 920
g = 9.81
theta = np.arange(0,1.18, 0.1)
theta = np.deg2rad(theta)


vol_list = H*L*W
    
A= (W/H)*(H**2)
  
 
As = (pi/pw)*A
A1 = (1/2)*((E*i)**2)*np.tan(theta) #Area of triangle dipped below surface
A2 = (A)*((pi/pw)-((1/2)*E*np.tan(theta))) #Area of submerged rectangle

tri_vol = A1*L
rect_vol= A2*L

C  = -H*np.cos(theta)*((pi/pw)-(1/2)) # icebergs center of mass
C1 = -(1/3)*(E)*H*np.sin(theta) #center of mass within triangle
C2 = -(H/2)*np.cos(theta)*((pi/pw)+(1/2)*E*(np.tan(theta))) #center of mass of the rectangle
Cs = (C1*A1 + C2*A2)/As #Center of Buoyancy
#Cs_26= -(1/2)*(pw/pi)*(H)*((E**2/12)*math.acos(theta)+((pi/pw)**2-(E**2/12))*np.cos(theta)) #Equations 26 Burton 2012 Center of Buoyancy

vol_s_list = tri_vol + rect_vol #volume of submerged/tilted iceberg

V= np.array(vol_list)
Vs= np.array(vol_s_list)

m = V*pi #mass of iceberg
ms= Vs*pw#mass of submerged portion of iceberg

#haf= h - H*(1- (pi/pw)) #height above/below floatation
#hab = np.arange(-4, 0, 0.0285)
Fg = m*(-g)
Fb = ms*(g)

Fg=np.array(Fg)
Fb=np.array(Fb)
Fnet = Fg + Fb #Net force
    
#T  = Fb*((H + haf)/2)*(np.sin(theta)) #torque moment
#
Sxx = (1/2)*(pi*g*H)*(1-((pw/pi)*(hdiff/H)**2)) #max shear stress
#Sxx= Sxx/1000 #kPa
area = W*H
#area = np.array(area)/1000000 
P = Fnet/area
#P = P/1000


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
