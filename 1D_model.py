import numpy as np 
import matplotlib.pyplot as plt
import sys
from math import sin, cos
from scipy.integrate import odeint

# generate initial conditions
phi = np.random.random(500)

L=5; B=0.01

def convolve1D(ar):
    convolved = np.zeros(len(ar))
    ar2 = np.concatenate([phi, phi]) # double array to prevent overflow indexing
    for i in range(len(ar)):
        convolved[i] = sum([np.exp(-(i-k)**2/(2*L**2))*ar2[k] for k in range(i-5*L, i+5*L)])
    return convolved
phi_conv = convolve1D(phi)

# define ramp-up function
def ramp(t):
    if t<=5:
        return t*2
    else:
        return ramp(5)
rampv = np.vectorize(ramp)

def Kappa(p):
    return (1-p)**3/p**2
Kappav = np.vectorize(Kappa)

def grad_p(x, t,w=1):
    return -(1-x)*ramp(t)*w/Kappa(x)

def d_phi_dt(phi, t):
    deriv = np.zeros(len(phi))
    for i, x in enumerate(phi):
        p = grad_p(x, t)
        if p**2 < B**2/L**2:
            deriv[i] = 0
        else:
            deriv[i] = -np.abs(x*(p**2 -B**2/L**2))
    return deriv

from scipy.integrate import odeint

steps = odeint(d_phi_dt, phi_conv, np.linspace(0,6, 1000))
#print(steps)

#plt.plot(np.linspace(0, 1, len(phi)), convolve1D(phi))
plt.style.use("fivethirtyeight")
fig, ax = plt.subplots(3,4, figsize=(10,6))
axs = ax.ravel()
for i, ax in enumerate(axs):
    ax.plot(np.linspace(0,1,500), steps[i*10]-phi_conv)
    ax.set_ylim(-2,0.1)
    ax.set_title(f"Erosion at t={i*10*6/1000}s", fontsize=5)

plt.tight_layout()
plt.suptitle("Erosion in 1D Model")

plt.savefig(f"plots/1D_erode_{L}_{B}.png")