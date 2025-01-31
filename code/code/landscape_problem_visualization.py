#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:37:26 2025

@author: itayta
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import cm


PATH = "/Users/itayta/Desktop/prot_stuff/results/illustrations/"



s = 4.5
non_smooth = False
random_heights = False
gen_rosetta_predictors = False
RH = 1.2
RUGGED_FACTOR = -.004

#rh_sample = (np.random.rand(len(multivariates) - 1) * 5)
rh_sample = np.array([2.986687  , 0.59792072, 1.42969075, 1.1938599 , 3.35635173, 0.68190825, 2.07049183]) # good seed

x, y = np.mgrid[-s:s:30j, -s:s:30j]
xy = np.column_stack([x.flat, y.flat])


multivariates = [[1, np.array([0.0, 0.0]), np.array([.75, .75])],
                 [1, np.array([-2.0, -2.0]), np.array([1., 1.])],
                 [-.18, np.array([-2.3, 1.1]), np.array([.5, .5])],
                 #[-.05, np.array([0.1, 0.1]), np.array([.15, .15])],
                 [.35, np.array([-3.5, 3.5]), np.array([.8, .8])],
                 [.075, np.array([.35, -2]), np.array([.48, .28])],
                 [-.38, np.array([2.9, -2.6]), np.array([.85, .95])],
                 [-.12, np.array([2.8, -.9]), np.array([.35, .35])],
                 [1, np.array([3.5, 3.5]), np.array([1.3, 1.3])]]                                                     
                                                         
                                      





idx = 0;

for conf in [["Smooth", True, False],
             ["Rugged", True, True],
             ["Rosetta_Predictor", False, False]]:
    m = multivariates[0]                   
    z =  m[0] * multivariate_normal.pdf(xy, mean=m[1], cov=np.diag(m[2])**2)



    title = conf[0]
    random_heights = conf[1]
    non_smooth = conf[2]
    
    if title == "Rosetta_Predictor":
        color_func = cm.magma
    else:
        color_func = cm.jet
    
    if gen_rosetta_predictors:
        rh_sample = (np.random.rand(len(multivariates) - 1) * 5)
        RH = (np.random.rand(1) * 2)[0]
        non_smooth = False
        random_heights = True
        title = "Gen_rosetta_%d" % idx
        idx += 1
        color_func = cm.magma
        

    for i, m in enumerate(multivariates[1:]):
        
        rh = RH
        
        if random_heights:
            rh = rh_sample[i]
        z +=  rh * m[0] * multivariate_normal.pdf(xy, mean=m[1], cov=np.diag(m[2])**2)
    
    
    i = 0
    
    if non_smooth:
        for xi in np.arange(-4,4, .5):
            for yi in np.arange(-4,4, .5): 
                print(xi)
                print(yi)
                i += 1
                factor = -1 if i % 2 ==0 else 1
                z += factor* RUGGED_FACTOR * multivariate_normal.pdf(xy, 
                                                    mean=np.array([xi, yi]), 
                                                    cov=np.diag(np.array([.05, .05])**2))
    
                                                                  
    z3 = z.reshape(x.shape)
    
    

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    my_col = color_func((z3 - np.amin(z3))/(np.amax(z3) - np.amin(z3)))
    ax.plot_surface(x,y,z3, facecolors = my_col, antialiased=False)
    
    #ax.plot_wireframe(x,y,z3)
    plt.savefig("%s/%s.pdf" % (PATH, title), format="pdf", bbox_inches="tight")
    plt.show()