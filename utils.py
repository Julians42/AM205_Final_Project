from math import tanh
import numpy as np
t_step =0.05


# ramp up 2D - push water in and suck it out
def fluid_forcing(pshape, t, scale, len_force=100, start=500):
    """Takes shape of initial phi matrix and time stamp
        -Returns forcing; tune magnitude of forcing with scale parameter"""
    f_force = np.zeros(pshape)
    # simple forcing - in in 1 location, out in another
    simple = np.ones(len_force)
    f_force[0][start:start+len_force] = -simple
    f_force[-1][start:start+len_force] = simple
    # define linear forcing as function of t (and scale)
    if t <5:
        return scale*t*f_force
    if t>=5:
        return scale*5*f_force

# function to calculate Kappa(phi)
def Kappa_phi(phi: np.ndarray) -> np.ndarray:
    one_mat = np.ones(phi.shape)
    return (one_mat - phi)**3/(phi**2)

# function to update fluid
def fluid_vel(phi, grad_p):
    return -np.multiply(Kappa_phi(phi), grad_p)

def psi(phi, c1 =0.5, w=5, phistar=0.4, c2=0.5):
    return c1*tanh(w*(phi-phistar))+c2

# function to calculate update rule dphi/dt
def dphi_FOFD(phi, grad_p, dt=t_step):
    dx = 1/phi.shape[0]; dy = 1/phi.shape[1]
    # compute stencils
    p_x = np.zeros(phi.shape); p_y = np.zeros(phi.shape)
    p_norm = np.zeros(phi.shape); psi_val = np.zeros(phi.shape)
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            p_x_update = ((grad_p[i+1, j]-grad_p[i,j])+(grad_p[i+1,j+1]-grad_p[i,j]))/(2*dx)
            p_y_update = ((grad_p[i,j+1]-grad_p[i,j])+(grad_p[i+1,j+1]-grad_p[i+1,j]))/(2*dy)
            p_x[i,j] = p_x_update; p_y[i,j]=p_y_update
            p_norm[i,j] = np.sqrt(p_x_update**2+p_y_update**2)
            psi_val[i,j] = psi(phi[i,j])
    # calculate gradient update
    f = np.maximum(np.zeros(phi.shape), p_norm -psi_val)
    phi_next = phi*(np.ones(phi.shape)- dt*f)
    return phi_next


# function to update grad p_n+1


# function to solve PDE