import numpy as np 
import matplotlib.pyplot as plt
import sys
import time
from math import sin, cos, pi, tanh
from random import seed
from numpy.random import random
from numpy import exp

from utils_2D import convolve_matrix, w_matrix, media_cmap

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
# what is the correct Kappa implementation

def Kappa(p):
    kp = (1-p)**3/(p**2)
    return min(.1, kp)
###########################
###########################

dim = int(np.ceil(res*(1+2*eps))) # expand range to get good communication length

phi_rand = random((dim, dim))
rho0 = np.ones((res,res))

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
def pressure(rho, c=c): # PLAY with c
    return c*(rho-1)

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

def flux_x(rho, phi, BC=np.zeros((rho0.shape[0],rho0.shape[1]+1)), dx =1/res, dt =dt):
    x_flux = np.zeros((rho0.shape[0],rho0.shape[1]+1))
    for row in range(x_flux.shape[0]):
        for column in range(1, x_flux.shape[1]-1):
            T1 = (rho[row, column-1]+rho[row, column])/2
            T2 = Kappa((phi[row, column-1]+phi[row, column])/2)
            T3 = (pressure(rho[row, column-1])-pressure(rho[row, column]))/dx
            x_flux[row, column] = T1*T2*T3
    return x_flux+dt*BC

def flux_y(rho, phi, BC=np.zeros((rho0.shape[0]+1,rho0.shape[1])), dy =1/res, dt=dt):
    y_flux = np.zeros((rho0.shape[0]+1,rho0.shape[1]))
    for row in range(1, y_flux.shape[0]-1):
        for column in range(y_flux.shape[1]):
            T1 = (rho[row-1, column]+rho[row, column])/2
            T2 = Kappa((phi[row-1, column]+phi[row, column])/2)
            T3 = (pressure(rho[row-1, column])-pressure(rho[row, column]))/dy
            y_flux[row, column] = T1*T2*T3
    num_bad = sum(sum(y_flux >1))
    if num_bad >1:
        print(num_bad)
    return y_flux + dt*BC


def update_rho(rho, phi, BC_x, BC_y, dt=dt, dx =1/res, dy = 1/res):
    # calculate mass flux at each boundary
    fx = flux_x(rho, phi, BC_x, dx=dx, dt=dt)
    fy = flux_y(rho, phi, BC_y, dy=dy, dt=dt)
    # initialize a flux matrix to get mass flux into a cell - this gives us an update for rho 
    flux_mat = np.zeros(rho.shape)
    for row in range(flux_mat.shape[0]):
        for column in range(flux_mat.shape[1]):

            fmat_x = (fx[row, column+1]+fx[row, column]) # flux in x-direction 
            fmat_y = (fy[row, column] + fy[row+1, column]) # flux in y-direction # Not mathematically correct but works physically
            flux_mat[row, column] = (fmat_x+fmat_y)
    # attempt to stop crazy fluid fields - apply a thresholding
    new_rho = rho+flux_mat
    new_rho = np.maximum(np.ones(new_rho.shape)/1.1, new_rho)
    new_rho = np.minimum(np.ones(new_rho.shape)/0.9, new_rho)

    return new_rho, fx, fy

def calc_pressure(rho_new, dx =1/res, dy=1/res):
    """Second order difference formulas gives the pressure"""
    # dx_mat = np.zeros(rho_new.shape)
    # dy_mat = np.zeros(rho_new.shape)
    p_mat = np.zeros(rho_new.shape)
    for row in range(rho_new.shape[0]):
        for column in range(rho_new.shape[1]):
            px = 0; py =0
            if column ==0: # forwards central difference
                px = (rho_new[row, column+2]-2*rho_new[row, column+1]+rho_new[row, column])/(dx**2)
            elif column == rho_new.shape[1]-1: # backwards
                px = (rho_new[row, column]-2*rho_new[row, column-1]+rho_new[row, column-2])/(dx**2)
            else: # centered 
                px = (rho_new[row, column+1]-2*rho_new[row, column]+rho_new[row, column-1])/(dx**2)
            if row ==0: # forwards central diff
                py = (rho_new[row+2, column]-2*rho_new[row+1, column]+rho_new[row, column])/(dy**2)
            elif row ==rho_new.shape[0]-1: # backwards
                py = (rho_new[row, column]-2*rho_new[row-1, column]+rho_new[row-2, column])/(dy**2)
            else:  # calculate second order centered difference
                py = (rho_new[row+1, column]-2*rho_new[row, column]+rho_new[row-1, column])/(dy**2)
            # calculate squared gradient of pressure 
            p_mat[row, column] = px**2+py**2
    return p_mat

# update phi based on changes to rho
def psi(phi, c1 =0.5, w=5, phistar=0.4, c2=0.5): # change stuff in HERE 
    return c1*tanh(w*(phi-phistar))+c2

def update_phi(phi, pressure, dt=dt, erode_factor=500):
    # get a derivative for phi
    dphi_dt = np.zeros(phi.shape)
    for i in range(dphi_dt.shape[0]):
        for j in range(dphi_dt.shape[1]):
            # based off pressure and function of solid fraction; always <=0
            dphi_dt[i,j] = - phi[i,j] * np.max([0, pressure[i,j]/erode_factor-psi(phi[i,j])]) # PLAY with dividing pressure
    # use forward Euler to update
    phi_next = phi + dt * dphi_dt
    # we can't have a negative solid fraction nor do we want a zero solid fraction because model blows up
    phi_next = np.maximum(phi_next, np.ones(phi_next.shape)/10)
    return phi_next


startTime = time.time()

# Implementation - step through algorithm.
phi_list = [fort]; rho_list =[rho0]; total_fluid = []
steps = int(sys.argv[1])
for i in range(steps):
    rho_next = update_rho(rho_list[i], smoothed, BC_x, BC_y)[0]
    total_fluid.append(sum(sum(rho_next)))
    pressure_current = calc_pressure(rho_next)
    phi_new = update_phi(phi_list[i], pressure_current, dt=dt, erode_factor=erode_factor)
    phi_list.append(phi_new); rho_list.append(rho_next)

# Calculate and print time
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))

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
plt.savefig("plots/wed_run_test.png", dpi=150)

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
plt.savefig("plots/total_fluid_volume.png", dpi=150)