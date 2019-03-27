# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:53:18 2019

@author: pc
"""

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

def lorenz(X, t, a, b, c):
    '''
    a = p[0], b = p[1], c = p[2]
    x = X[0], y = X[1], z = X[2]
    dx/dt = a*(y - x)
    dy/dt = x*(b - z) - y
    dz/dt = x*y - c*z
    
    '''
    
    # a = p[0], b = p[1], c = p[2]
    (x, y, z) = X
    dx = a*(y - x)
    dy = x*(b - z) - y
    dz = x*y - c*z
    
    return np.array([dx, dy, dz])
t = np.linspace(0,20, 10000)
lorenz_state = odeint(lorenz, (0, 1, 0), t, args = (10, 28, 3))
plt.plot(t, lorenz_state[:,0])
plt.plot(t, lorenz_state[:,1])
plt.plot(t, lorenz_state[:,2])
# plt.show()