#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
<<<<<<< HEAD
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

=======
Created on Mon Mar  2 10:08:21 2020

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
>>>>>>> 2f4f0f9f396a45255015b56b0c82b37d044ad2b1
os.chdir('/data/stor/basic_data/spatial_data/greenland/bedmap/')
filename = 'BedMachineGreenland-2017-09-20.nc'

file = nc.Dataset(filename)

ice_thickness = file['thickness'] #meters
bed_elevation = file['bed'] #meters
mask = file['mask'] #0 = ocean, 1 = ice-free land, 2 = grounded ice, 3 = floating ice, 4 = non-Greenland land
sur = file['surface']

<<<<<<< HEAD
# Array of ice thicknesses at terminus and corresponding surface elevations

iceth = np.array(ice_thickness[8960:9020,2780:2840]) #ice thickness array at Rink
ice_block = iceth[27:33,33:36]                       #smaller ice thickness array
unice = ice_block.ravel()                            #list of ice thicknesses near front of ice tooth
surth = np.array(sur[8960:9020,2780:2840])           #surface elevation array at Rink
sur_block = surth[27:33,33:36]                       #smaller surface elevation array
unsur = sur_block.ravel()                            #list of surface elevations near front of ice tooth

# Hard Force Balance Variables

L = 1340                                             #Length of iceberg perpendicular to flow (m)
W_t = 340                                             #Width of iceberg at tooth parallel to flow (m)
W_s = 230                                            #Width of iceberg at rotating shadow in MLI(m)
pw = 1020                                            #density of water (kg/m**2)
pi = 920                                             #density of ice (kg/m**2)
g = 9.81                                             #gravitational acceleration (m/s**2)

H= np.arange(450, 700, 10)                           #estimated iceberg heights from BedMachine (m)
E = W_s/H                                              #iceberg aspect ratio

# Angles on glacier REFER TO IDEAL ICEBERG IN THESIS (FIG. 8)

geoid_H = 31.657                                     # geoid height at rink (m)

#H = np.arange(450, 700, 10)

a = 427                                              #TRI elevation (m)
b = 63 - geoid_H                                     #upglacier elevation (m)
c = 54 - geoid_H                                     #arcticdem terminus elevation (m)

ab = 5700                                            #distance between TRI and glacier (m)
bc = 75                                              #horizontal distance between c and b (m)

alpha = math.atan((a-b)/ab)                          #angle b/w TRI and horizontal
alpha = np.rad2deg(alpha)
    
beta= math.atan((b-c)/bc)                            #angle b/w flowing glacier and horizontal
beta = np.rad2deg(beta)

delta = alpha + beta                                 #angle b/w flowing glacier and TRI

B = math.sin(np.deg2rad(delta))*bc                   #elevation between iceberg hinge and top of radar gaze angle
y = 180 - 90- delta                                  #angle next to delta at hinge

zeta = np.rad2deg(math.acos(B/W_s))                 #angle complimentary to lowercase gamma (y) at hinge
theta = 180 - y - zeta                              # Rotation angle of iceberg 
psi = theta - beta                                  #angle between horizontal seal level and iceberg rotating                                            

#Additional Force Variables (Iceberg lengths, Volume, Area, Mass)

C1 = math.tan(np.deg2rad(theta))*(W_s)              #length of iceberg facing the ocean tipping out of the water (Fig 8 - C)
C2 = math.tan(np.deg2rad(psi))*(W_s)                #length of iceberg used to separate the aerial portions of iceberg (Fig 8 - C)
C3 = C1-C2                                          #inner length of the aerial portion of iceberg

 
A= H*W_s                                            #area of idealized iceberg face (Fig 8)
A1 = (1/2)*C2*W_s                                   #area of triangle rotated above ocean surface            
A2 = (C3)*W_s                                       #area of rectangle rotated above ocean surface
Vi = L*W_s*H                                        #volume of ice with different heights
Va = A1*L + A2*L                                    #volume of iceberg above water
Vs = Vi - Va                                        #volume of submerged ice with different heights

m = Vi*pi                                           #mass of iceberg
ms= Vs*pw                                           #mass of submerged portion of iceberg

Fg = -pi*g*Vi                                       #force of gravity for different masses of icebergs
Fb = pw*g*Vs                                        #force of buoyancy for different masses of icebergs

Fnet = -(Fg + Fb)                                   #net force on iceberg

Pmax = Fnet/A[1]                                    #shear strength (Pa)
Pmax = Pmax/1000                                    # (kPa)

# Plot shear strength b/w tooth and 3rd iceberg

plt.figure()
plt.plot(H, Pmax,  '.')
plt.grid()

plt.ylabel("Shear Strength (kPa)")
plt.xlabel('Terminus thickness (m)')

plt.gcf().autofmt_xdate()

# Torque moment due to buoyancy

cm  = -H*np.cos(theta)*((pi/pw)-(1/2))           #iceberg's center of mass
cm1 = -(1/3)*(E)*H*np.sin(theta)                 #center of mass within triangle
cm2 = -(H/2)*np.cos(theta)*((pi/pw)+(1/2)*E*(np.tan(theta))) #center of mass of the rectangle
cm_s = (C1*A1 + C2*A2)/(pi/pw)*A                  #Center of Buoyancy


T = g*pi*L*A*(cm_s-cm)

plt.figure()
plt.plot(T, E,  '.')
plt.xlabel("Torque moment (N/m)")
plt.ylabel('Aspect Ratio')


plt.gcf().autofmt_xdate()

=======
#%%
plt.figure()
plt.imshow(ice_thickness[8960:9020,2780:2840]) #[2813.83,8986.85]
plt.figure()
plt.imshow(sur[8960:9020,2780:2840])


plt.plot([62.54],[115.2],'o', color='red', markersize = 6)  

#%% Array of ice thicknesses at terminus and corresponding surface elevations
iceth = np.array(ice_thickness[8960:9020,2780:2840])
ice_block = iceth[27:33,33:36]
unice = ice_block.ravel() #list of ice thicknesses
surth = np.array(sur[8960:9020,2780:2840])
sur_block = surth[27:33,33:36]
unsur = sur_block.ravel() #list of surface elevations
D= np.array(bed_elevation[8960:9020,2780:2840])
D = D[27:33,33:36]
unD = D.ravel() #list of waterdepths 

floating= surth/iceth #if 0.10 ice is at neutral buoyancy, if below then its below buoyancy, if above than greater than neutral buoyancy
plt.imshow(floating, vmin = 0, vmax = 0.2)
plt.colorbar()
plt.title('nearness to neutral buoyancy (0.1)')
#%% Force Variables

L= 1420 #m
W=  400 #m

H= unice
E = W/H 
h = unsur
hdiff= (H-h)
pw = 1020
pi = 920
g = 9.81
theta = np.deg2rad(1.18)

vol_list =[] #whole iceberg
for i in H:
    vol = L*W*i
    vol_list.append(vol)
    
A=[]        #Area of idealized iceberg
for i in H:
    a = (W/i)*(i**2)
    A.append(a)
A = np.array(A)   
 
As = (pi/pw)*A
A1 = (1/2)*((E*i)**2)*np.tan(theta) #Area of triangle dipped below surface
A2 = (A)*((pi/pw)-((1/2)*E*np.tan(theta))) #Area of submerged rectangle

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
>>>>>>> 2f4f0f9f396a45255015b56b0c82b37d044ad2b1
