import numpy as np
from math import pi, tanh

##############################
##############################

# generating a smooth field matrix
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


def convolve_matrix(phi, eps, res):
    """Smoothing function: applies weight matrix to all gridpoints in phi matrix"""
    eps_res = int(eps*res)
    smoothed = np.zeros((res, res)) # initiallize
    # generate weight matrix for this resolution plot
    weight_matrix = w_matrix(eps, res)
    for i in range(eps_res, smoothed.shape[0]+eps_res):
        for j in range(eps_res,smoothed.shape[0]+eps_res):
            # index out communication area
            communication_area = phi[i-eps_res:i+eps_res, j-eps_res: j+eps_res]
            # elementwise sum of weights and communication points
            smoothed[i-eps_res,j-eps_res] = np.sum(np.multiply(communication_area, weight_matrix))/((eps_res*2+1))**2
    
    return smoothed/(np.max(smoothed))

##############################
##############################
# Kappa helper function 
def Kappa(p):
    return (1-p)**3/(p**2)

# updating fluid flux
eps = 0.05; res = 200 # defaults
dim = int(np.ceil(res*(1+2*eps)))
def q_x(rho, phi, dx = 1/dim):
    """Calculates flux on the horizonal edges"""
    q_xupdate = np.zeros((phi.shape[0]+1, phi.shape[1]))
    for row in range(1, phi.shape[0]):
        for column in range(phi.shape[1]):
            T1 = (rho[row-1,column]+rho[row,column])/2
            T2 = Kappa((phi[row-1,column]+phi[row-1,column])/2)
            T3 = (rho[row-1, column]+rho[row,column])/dx
            q_xupdate[row, column] = T1 * T2 * T3
    return q_xupdate

def q_y(rho, phi, dy =1/dim):
    """Calculates flux on the vertical edges"""
    q_yupdate = np.zeros((phi.shape[0], phi.shape[1]+1))
    # iterate through edges
    for row in range(phi.shape[0]):
        for column in range(1, phi.shape[1]):
            # compute based on phi and rho of neighboring points
            T1 = (rho[row, column]+rho[row, column-1])/2
            T2 = Kappa((phi[row, column]+phi[row, column-1])/2)
            T3 = (rho[row, column]-rho[row, column-1])/dy
            q_yupdate[row, column] = T1 * T2 * T3
    return q_yupdate

##############################
##############################

def update_rho(qx, qy, rho, dx =1/dim, dy=1/dim, dt =0.5, c=0.1**2):
    tflux = np.zeros(rho.shape)
    for row in range(tflux.shape[0]):
        for column in range(tflux.shape[1]):
            tflux[row, column] = (qx[row, column]-qx[row+1, column])/dx + (qy[row, column+1]-qy[row, column])/dy
            #tflux[row, column] = rho[row, column] - qx[]

    # update with euler step
    rho_updated = (rho+c*dt*tflux)
    # normalize update matrix - prevents fluid creation/destruction 
    norm_factor=  np.sum(rho)/np.sum(rho_updated)
    return rho_updated*norm_factor

def rho_np1(phi, rho, dim, dt = .5, c=0.005**2):
    # calculate flux in x and y directions
    flux_x = q_x(rho, phi, dx=1/dim)
    flux_y = q_y(rho, phi, dy=1/dim)

    # update rho based on fluxes
    rho_next = update_rho(flux_x, flux_y, rho, dx=1/dim, dy=1/dim, dt=dt, c=c)

    return rho_next

##############################
##############################

# update phi based on changes to rho
def psi(phi, c1 =0.5, w=5, phistar=0.4, c2=0.5):
    return c1*tanh(w*(phi-phistar))+c2

def update_phi(phi, rho, dt=0.05):
    # get a derivative for phi
    dphi_dt = np.zeros(phi.shape)
    for i in range(dphi_dt.shape[0]):
        for j in range(dphi_dt.shape[1]):
            # based off pressure and function of solid fraction; always <=0
            dphi_dt[i,j] = - phi[i,j] * np.max([0, rho[i,j]**2-psi(phi[i,j])])
    # use forward Euler to update
    phi_next = phi + dt * dphi_dt
    # we can't have a negative solid fraction - we set this to zero
    phi_next = np.maximum(phi_next, np.zeros(phi_next.shape))
    return phi_next