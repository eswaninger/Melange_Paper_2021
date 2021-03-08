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

# Hard Force Balance Variables

L = 1340                                             #Length of iceberg perpendicular to flow (m)
W =  340                                             #Width of iceberg at tooth parallel to flow (m)
W_s = 230                                            #Width of iceberg at rotating shadow in MLI(m)
pw = 1020                                            #density of water (kg/m**2)
pi = 920                                             #density of ice (kg/m**2)
g = 9.81                                             #gravitational acceleration (m/s**2)

H= unice                                             #estimated iceberg heights from BedMachine
E = W/H                                              #iceberg aspect ratio

haf= unsur - H*(1- (pi/pw))                          #height above/below floatation (m)

# Angles on glacier REFER TO IDEAL ICEBERG IN THESIS (FIG. 8)

geoid_H = 31.657                                     #m

#H = np.arange(450, 700, 10)

a = 427                                              #TRI elevation
b = 63 - geoid_H                                     #upglacier elevation
c = 54 - geoid_H                                     #arcticdem terminus elevation

ab = 5700                                            #distance between TRI and glacier 
bc = 75                                              #horizontal distance between c and b

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

#Additional Force Variables (Volume, Area, Mass)

theta = np.deg2rad(theta)

C1 = math.tan(np.deg2rad(theta))*(W_s)
C2 = math.tan(np.deg2rad(psi))*(W_s)
C3 = C1-C2

vol_list = []                                        #whole iceberg estimated volume
for i in H:
    vol = L*W*i
    vol_list.append(vol)
    
A=[]                                                #area of idealized iceberg
for i in H:
    a = (W_s/i)*(i**2)
    A.append(a)
A = np.array(A)   
 
A= H*W_s     
A1 = (1/2)*C2*W_s                                   #area of triangle rotated above ocean surface            
A2 = (C3)*W_s                                       #area of rectangle rotated above ocean surface
Vi = L*W_s*H                                        #volume of ice with different heights
Va = A1*L + A2*L                                    #volume of iceberg above water
Vs = Vi - Va                                        #volume of submerged ice with different heights

m = Vi*pi                                            #mass of iceberg
ms= Vs*pw                                           #mass of submerged portion of iceberg

Fg = -pi*g*Vi                                      #force of gravity for different masses of icebergs
Fb = pw*g*Vs                                        #force of buoyancy for different masses of icebergs

Fnet = -(Fg + Fb)                                      #net force on iceberg

Pmax = Fnet/A [1]                                      #shear strength (Pa)
Pmax = Pmax/1000                                    # kPa

# Plot shear strength b/w tooth and 3rd iceberg

plt.figure()
plt.plot(A, Pmax,  '.')
plt.grid()

plt.ylabel("Shear Strength (kPa)")
plt.xlabel('area (m**2)')

plt.gcf().autofmt_xdate()

# Torque moment due to buoyancy

C  = -H*np.cos(theta)*((pi/pw)-(1/2))           #iceberg's center of mass
C1 = -(1/3)*(E)*H*np.sin(theta)                 #center of mass within triangle
C2 = -(H/2)*np.cos(theta)*((pi/pw)+(1/2)*E*(np.tan(theta))) #center of mass of the rectangle
Cs = (C1*A1 + C2*A2)/(pi/pw)*A                  #Center of Buoyancy


T = g*pi*L*A*(Cs-C)

plt.figure()
plt.plot(T, E,  '.')
plt.xlabel("Torque moment (N/m)")
plt.ylabel('Aspect Ratio')


plt.gcf().autofmt_xdate()

#%%
plt.figure()
plt.plot(C2,Cs)

fig, axes = plt.subplots(nrows= 4, clear = True)
y = np.arange(0,60, 1)
axes[0].plot(theta, Pmax, '.', color='blue', markersize = 3.5)
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
