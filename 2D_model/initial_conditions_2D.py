import numpy as np
from numpy import exp
from math import pi
import matplotlib.pyplot as plt
from random import seed

seed(205) # set seed 
# define constants and initialize grid
eps = 0.05
res=200
dim = int(np.ceil(res*(1+2*eps)))
phi = np.random.rand(dim,dim)


def w_matrix(eps, res):
    """Defines exponential decay matrix based on grid resolution (res) and communication length eps"""
    eps_res = 2*int(eps*res) # eps on each side
    weight_matrix = np.zeros((eps_res, eps_res))
    i, j = np.floor(weight_matrix.shape[0])/2, np.floor(weight_matrix.shape[0])/2 # index of middle gridpoint
    # iterate through gridpoints to determine weighting by distance
    for k in range(weight_matrix.shape[0]):
        for l in range(weight_matrix.shape[1]):
            # division (can be adjusted) gives scale of how fast weighting decreases
            weight_matrix[k,l] = np.exp(- (((i-k)/res)**2+((j-l)/res)**2) /(eps**2/pi)) 
    return weight_matrix
weight_matrix = w_matrix(eps, res)

def convolve_matrix(phi, eps, res):
    """Smoothing function: applies weight matrix to all gridpoints in phi matrix"""
    eps_res = int(eps*res)
    smoothed = np.zeros((res, res))
    for i in range(eps_res, smoothed.shape[0]+eps_res):
        for j in range(eps_res,smoothed.shape[0]+eps_res):
            # index out communication area
            communication_area = phi[i-eps_res:i+eps_res, j-eps_res: j+eps_res]
            # elementwise sum of weights and communication points
            smoothed[i-eps_res,j-eps_res] = np.sum(np.multiply(communication_area, weight_matrix))/((eps_res*2+1))**2
    
    return smoothed/(np.max(smoothed))

smoothed = convolve_matrix(phi, eps, res)
plt.imshow(smoothed)
plt.colorbar()
plt.show()