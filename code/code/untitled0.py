#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 12:24:34 2024

@author: itayta
"""





# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
 
# Creating dataset
x = np.outer(np.linspace(-3, 3, 32), np.ones(32))
y = x.copy().T # transpose
z = (np.sin(.3  * x **2) + np.cos(.3  * y **2) )
 
# Creating figure
fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')
 
# Creating color map
my_cmap = plt.get_cmap('hot')
 
# Creating plot
surf = ax.plot_surface(x, y, z,
                       cmap = plt.cm.Blues_r,
                       edgecolor ='black')
 
fig.colorbar(surf, ax = ax,
             shrink = 0.5, aspect = 5)
 
ax.set_title('Surface plot')
 
# show plot
plt.show()