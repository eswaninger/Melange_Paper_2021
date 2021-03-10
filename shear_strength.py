#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

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


# Data

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
