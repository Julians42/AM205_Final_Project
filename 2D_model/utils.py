import numpy as np
from math import pi, tanh


eps =0.1; res=100
rho0 = np.ones((res,res))
V=1 # set v order 1
dt = 1/2*(1/res)*V/10 # dt based on the CFL condition  - PLAY
c = 1 # pressure coefficient



def Kappa(p):
    """Function of Solid Fraction"""
    kp = (1-p)**3/(p**2)
    return min(.1, kp)

def pressure(rho, c=c): # PLAY with c
    return c*(rho-1)

def flux_x(rho, phi, BC=np.zeros((rho0.shape[0],rho0.shape[1]+1)), dx =1/res, dt =dt):
    """Calculates flux in the x-direction"""
    x_flux = np.zeros((rho0.shape[0],rho0.shape[1]+1))
    for row in range(x_flux.shape[0]):
        for column in range(1, x_flux.shape[1]-1):
            T1 = (rho[row, column-1]+rho[row, column])/2
            T2 = Kappa((phi[row, column-1]+phi[row, column])/2)
            T3 = (pressure(rho[row, column-1])-pressure(rho[row, column]))/dx
            x_flux[row, column] = T1*T2*T3
    return x_flux+dt*BC

def flux_y(rho, phi, BC=np.zeros((rho0.shape[0]+1,rho0.shape[1])), dy =1/res, dt=dt):
    """Calculates flux in the y-direction"""
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
