import numpy as np
from numpy import exp
from math import pi
import matplotlib.pyplot as plt
from numpy.core.numeric import convolve

eps = 0.05
res=1000
dim = int(np.ceil(res*(1+2*eps)))
phi = np.random.rand(dim,dim)


def w_matrix(eps, res):
    eps_res = 2*int(eps*res)
    weight_matrix = np.zeros((eps_res, eps_res))
    i, j = np.floor(weight_matrix.shape[0])/2, np.floor(weight_matrix.shape[0])/2
    for k in range(weight_matrix.shape[0]):
        for l in range(weight_matrix.shape[1]):
            weight_matrix[k,l] = np.exp(- (((i-k)/res)**2+((j-l)/res)**2) /(2*np.pi*eps**2))
    return weight_matrix
weight_matrix = w_matrix(eps, res)

def convolve_matrix(phi, eps, res):
    eps_res = int(eps*res)
    smoothed = np.zeros((res, res))
    for i in range(eps_res, smoothed.shape[0]+eps_res):
        for j in range(eps_res,smoothed.shape[0]+eps_res):
            communication_area = phi[i-eps_res:i+eps_res, j-eps_res: j+eps_res]
            smoothed[i-eps_res,j-eps_res] = np.sum(np.multiply(communication_area, weight_matrix))/(eps_res*2+1)**2
    return smoothed

smoothed = convolve_matrix(phi, eps, res)
plt.imshow(smoothed)
plt.colorbar()
plt.show()