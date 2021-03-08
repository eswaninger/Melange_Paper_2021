#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
<<<<<<< HEAD
Created on Wed Feb 24 10:55:26 2021

@author: eswaninger

This code computes the shear strength of ice on the terminus

"""
#%% Import Modules

import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import os
import math

"""TEST TO SEE IF I CAN COMMIT THIS"""

#%%

os.chdir('/data/stor/basic_data/spatial_data/greenland/bedmap/')
filename = 'BedMachineGreenland-2017-09-20.nc'

file = nc.Dataset(filename)

ice_thickness = file['thickness'] #meters
bed_elevation = file['bed'] #meters
mask = file['mask'] #0 = ocean, 1 = ice-free land, 2 = grounded ice, 3 = floating ice, 4 = non-Greenland land
sur = file['surface']


# Array of ice thicknesses at terminus and corresponding surface elevations

iceth = np.array(ice_thickness[8960:9020,2780:2840]) #ice thickness array at Rink
ice_block = iceth[27:33,33:36]                       #smaller ice thickness array
unice = ice_block.ravel()                            #list of ice thicknesses near front of ice tooth
surth = np.array(sur[8960:9020,2780:2840])           #surface elevation array at Rink
sur_block = surth[27:33,33:36]                       #smaller surface elevation array
unsur = sur_block.ravel()                            #list of surface elevations near front of ice tooth

# Hard force Balance Variables

L= 1420                                              #m Length of iceberg perpendicular to flow
W=  400                                              #m Width of iceberg parallel to flow
pw = 1020                                            #density of water
pi = 920                                             #density of ice
g = 9.81                                             #gravitational acceleration

H= unice                                             #estimated iceberg heights
E = W/H                                              #iceberg aspect ratio

#Additional Force Variables (Volume, Area, Mass)

theta = np.deg2rad(1.18)

vol_list =[]                                        #whole iceberg
for i in H:
    vol = L*W*i
    vol_list.append(vol)
    
A=[]                                                #area of idealized iceberg
for i in H:
    a = (W/i)*(i**2)
    A.append(a)
A = np.array(A)   
 
As = (pi/pw)*A
A1 = (1/2)*((E*i)**2)*np.tan(theta)                 #area of triangle dipped below ocean surface
A2 = (A)*((pi/pw)-((1/2)*E*np.tan(theta)))          #area of submerged rectangle

tri_vol = A1*L
rect_vol= A2*L

C  = -H*np.cos(theta)*((pi/pw)-(1/2)) # icebergs center of mass
C1 = -(1/3)*(E)*H*np.sin(theta) #center of mass within triangle
C2 = -(H/2)*np.cos(theta)*((pi/pw)+(1/2)*E*(np.tan(theta))) #center of mass of the rectangle
Cs = (C1*A1 + C2*A2)/As #Center of Buoyancy
Cs_26= -(1/2)*(pw/pi)*(H)*((E**2/12)*math.acos(theta)+((pi/pw)**2-(E**2/12))*np.cos(theta)) #Equations 26 Burton 2012 Center of Buoyancy

vol_s_list = tri_vol + rect_vol #volume of submerged/tilted iceberg

V= np.array(vol_list)
Vs= np.array(vol_s_list)

m = V*pi #mass of iceberg
ms= Vs*pw#mass of submerged portion of iceberg

#haf= h - H*(1- (pi/pw)) #height above/below floatation
#hab = np.arange(-4, 0, 0.0285)
Fg = []
for i in m:
    F = i*(g)
    Fg.append(F)
Fb =[]
for i in ms:
    F = i*(-g)
    Fb.append(F)

Fg=np.array(Fg)
Fb=np.array(Fb)
Fnet = Fg + Fb #Net force
    
#T  = Fb*((H + haf)/2)*(np.sin(theta)) #torque moment
#
Sxx = (1/2)*(pi*g*H)*(1-((pw/pi)*(hdiff/H)**2)) #max shear stress
Sxx= Sxx/1000
area=[]
for i in H:
    Ac = W*i
    area.append(Ac)
area = np.array(area)
P = Fnet/area
P = P/1000

#%% Plot Shear strength bw tooth and 3rd iceberg
plt.figure()
plt.plot(area, P,  '.')
plt.grid()

#plt.ylabel("Shear Strength (Pa)")
#plt.xlabel('area (m**2)')

#%% Torque moment due to buoyancy

T = g*pi*L*A*(Cs-C)

plt.figure()
plt.plot(T, E,  '.')
plt.xlabel("Torque moment (N/m)")
plt.ylabel('Aspect Ratio')


#%%

fig, axes = plt.subplots(nrows= 4, clear = True)
y = np.arange(0,60, 1)
axes[0].plot(E, P, '.', color='blue', markersize = 3.5)
#axes[0,0].plot(np.unique(area), np.poly1d(np.polyfit(area, P, 2))(np.unique(P)))
axes[0].set_ylabel('Shear Strength (kPa)', fontsize = 12)
axes[0].yaxis.set_label_coords(-0.09,0.5)
axes[0].tick_params(axis='x', which='major', labelsize=12)
axes[0].tick_params(axis='y', which='major', labelsize=12)
axes[0].grid() 
axes[0].axvspan(0.75,0.85, alpha= 0.2, color = 'gray')

axes[1].plot(E,Sxx,'.', color='blue',markersize = 3.5)
axes[1].set_ylabel('Sxx (kPa)', fontsize = 12)
axes[1].yaxis.set_label_coords(-0.09,0.5)
axes[1].tick_params(axis='x', which='major', labelsize=12)
axes[1].tick_params(axis='y', which='major', labelsize=12)
axes[1].grid() 
axes[1].axvspan(0.75,0.85, alpha= 0.2, color = 'gray')

axes[2].plot(E,Fnet, '.',color='blue', markersize =3.5)
axes[2].set_ylabel('Net-Force (N)', fontsize = 12)
axes[2].yaxis.set_label_coords(-0.09,0.5)
axes[2].tick_params(axis='x', which='major', labelsize=12)
axes[2].tick_params(axis='y', which='major', labelsize=12)
axes[2].grid() 
axes[2].axvspan(0.75,0.85, alpha= 0.2, color = 'gray')

axes[3].plot(E,T, '.',color='blue', markersize =3.5)
axes[3].set_ylabel('Torque Moment (Nm)', fontsize = 12)
axes[3].yaxis.set_label_coords(-0.09,0.5)
axes[3].tick_params(axis='x', which='major', labelsize=12)
axes[3].tick_params(axis='y', which='major', labelsize=12)
axes[3].grid() 
axes[3].axis('auto')
axes[3].set_xlabel("Aspect Ratio", fontsize=12)
axes[3].axvspan(0.75,0.85, alpha= 0.2, color = 'gray')


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
Sxx= Sxx/1000 #kPa
area = W*H
#area = np.array(area)/1000000 
P = Fnet/area
P = P/1000

T = g*pi*L*A*(Cs-C) #torque at buoyancy

t = np.arange(0,12,1)


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
