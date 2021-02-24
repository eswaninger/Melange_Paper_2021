#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 09:45:33 2019

@author: eswaninger
"""
#%%
''' This script is a tester for solving for true flow velocities connected to 
openPIV velocity maps that solved for true flow direction using MLI_RECT files. 
'''

#%%
import numpy as np
import matplotlib.pyplot as plt
import math
#%%
file = '20140729_044000u_20140729_044230u.adf.unw.rec'
path= '/data/stor/basic_data/tri_data/rink/proc_data/d0728/INT/REC/'

file_path = (path + file)

slopes=[]
alphas=[]

yr = 778
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
beta = 202 #degrees from 0 north (),
alpha = 360 - beta - gamma
alpha = np.reshape(alpha, (1559,845))

with open(file_path,'rb') as f:
    temp= f.read()
    pha_= np.fromfile(file_path, dtype='>f')
    pha_[pha_==0] = np.nan
    pha_rectangle = np.reshape(pha_, (1559,845))#[600:630,755:785]
    vlos = (-0.0175*pha_rectangle)/(4* 3.14159*(2.5/1440))
    unw = vlos #[100:350, 366:570]
    flow = vlos/(np.cos(np.radians(alpha)))

plt.imshow(flow)
