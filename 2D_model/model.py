import numpy as np 
import matplotlib.pyplot as plt
import sys
import time
from math import sin, cos, pi, tanh
from random import seed
from numpy.random import random
from numpy import exp

# local imports - across 2 separate files
from utils_2D import convolve_matrix, w_matrix, media_cmap
from utils import *

###########################
###########################
# Params to play with
eps =0.1; res=100
V=1 # set v order 1
dt = 1/2*(1/res)*V/10 # dt based on the CFL condition  - PLAY
c = 1 # pressure coefficient
erode_factor = 50; scale_init=1
erode_factor = 50000; scale_init=1
gap=5
###########################
###########################

dim = int(np.ceil(res*(1+2*eps))) # expand range to get good communication length

phi_rand = random((dim, dim))

# boundary conditions
BC_y = np.zeros((rho0.shape[0]+1,rho0.shape[1]))
start = int(res/2)
BC_y[0, start:start+gap] = np.ones(gap)/2
BC_y[-1, start:start+gap] = -np.ones(gap)/2

BC_x = np.zeros((rho0.shape[0], rho0.shape[1]+1))
# BC_x[10:15, 0] = np.ones(5)/2
# BC_x[10:15, -1] = -np.ones(5)/2

# apply smoothing
smoothed = convolve_matrix(phi_rand, eps, res)/scale_init
print(np.max(np.max(smoothed)))

###########################
###########################
# Dam Implementation
fort = smoothed.copy()
fstart = int(res/2-res/10); fend = int(res/2+res/10)
fgap = int(res/20)
fort[fstart:fend, fstart:fend] = np.ones((fend-fstart, fend-fstart))
fort[fstart+fgap:fend-fgap, fstart+fgap:fend-fgap] = smoothed[fstart+fgap:fend-fgap, fstart+fgap:fend-fgap]
###########################
###########################

##### COMPUTATION #########
startTime = time.time()

# Implementation - step through algorithm.
phi_list = [fort]; rho_list =[rho0]; total_fluid = []
steps = int(sys.argv[1])
for i in range(steps):
    # Iterate time steps
    rho_next = update_rho(rho_list[i], smoothed, BC_x, BC_y)[0]
    total_fluid.append(sum(sum(rho_next)))
    pressure_current = calc_pressure(rho_next)
    phi_new = update_phi(phi_list[i], pressure_current, dt=dt, erode_factor=erode_factor)
    phi_list.append(phi_new); rho_list.append(rho_next)

# Calculate and print time
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))

###########################
###########################

# Make some nice plots
plt.style.use("fivethirtyeight")
fig, ax = plt.subplots(3,3, figsize=((10,10)))

plot_step = int(steps/10)
for i, axs in enumerate(ax.flat):
    axs.set_axis_off()
    im = axs.imshow(phi_list[plot_step*i], cmap=media_cmap, vmin=0.1, vmax=1)
    axs.set_title(f"Erosion at {round(i*plot_step*dt,3)}s")
plt.suptitle("Erosion in Porous Media")
cb_ax = fig.add_axes([0.94, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cb_ax)
cbar.set_label("Solid Fraction")
#plt.savefig("plots/wed_run_test.png", dpi=150)

plt.clf()

fig, ax = plt.subplots(3,3, figsize=((10,10)))

axs = ax.ravel()

for i, axs in enumerate(ax.flat):
    axs.set_axis_off()
    im = axs.imshow(rho_list[plot_step*i], cmap=media_cmap, vmin=0.75, vmax=1.25)
    axs.set_title(f"Fluid Pressure: {round(i*plot_step*dt,3)}s")
plt.suptitle("Fluid Pressure in Porous Media")
cb_ax = fig.add_axes([0.94, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cb_ax)
cbar.set_label("Fluid Fraction")
plt.savefig("plots/wed_run_test_rho.png", dpi=150)

plt.clf()

# plot total fluid 
xs = np.linspace(0, steps*dt, len(total_fluid))

plt.plot(xs, np.array(total_fluid)/(res**2), alpha=0.6, color="red", linewidth=4)
plt.title("Total Fluid Volume")
plt.xlabel("Time (s)")
plt.ylabel("Fluid Volume (norm by initial conditions")
#plt.savefig("plots/total_fluid_volume.png", dpi=150)