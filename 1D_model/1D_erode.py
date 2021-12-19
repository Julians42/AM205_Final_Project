import numpy as np 
import matplotlib.pyplot as plt
import sys
from math import sin, cos
from scipy.integrate import odeint

L=4
# generate initial conditions
phi = np.random.random(500)

# convolve x to get initial conditions
def convolve(ar, L=L):
    convolved = np.zeros(len(phi))
    for ind, x in enumerate(ar):
        convolved[ind] = sum([   np.exp( (ind-i)**2/(2*L**2)) *ar[i] for i in range(len(ar))])
    return convolved

print(convolve(phi))