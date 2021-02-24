#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:05:15 2021

@author: eswaninger
"""

#%%

import numpy as np
import matplotlib.pyplot as plt

#%%



file = '20140728_130230u_20140728_130500u.adf_rec.diff.bmp'
path = '/data/stor/basic_data/tri_data/temp/'

file_path = (path + file)

with open(file_path, 'rb') as f:
    temp = f.read()
    temperature = np.fromfile(file_path, dtype= '>f')
    temperature[temperature==0] = np.nan
    temp_rectangle = np.reshape(temperature, (2243,1348))
    
plt.figure()
plt.imshow(temp_rectangle)
    