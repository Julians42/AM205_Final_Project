from utils import *
# from initial_conditions_2D import eps, res, dim, phi, weight_matrix, convolve_matrix
import matplotlib.pyplot as plt
from math import pi

res = 50
eps =0.2
dim = int(np.ceil(res*(1+2*eps)))
# Constants
t_step = 0.05
dx = 1/res; dy = 1/res # assume grid is total size 1
weight_matrix = w_matrix(eps, res)
phi0 = convolve_matrix(np.random.rand(dim, dim), eps, res)


# generate initial conditions
#phi0 = convolve_matrix(phi, eps, res)
rho0 = np.zeros(phi0.shape)
q0_x = np.zeros((phi0.shape[0]+1, phi0.shape[1]))

# fluid forcing on the x_direction of q
additive_q = fluid_forcing_x(q0_x.shape, t_step*1, 1, len_force=int(res/4), start=int(res/2)) # fluid forcing on q

# we now derive an update rule for q based on phi and rho
def update_rho(phi_mat, rho_mat, Kappa, add_qx_mat, add_qy_mat, dx =dx):
    """Updates pressure matrix by considering flux"""
    # have one more in the direction of interrest because of boundary conditions
    q_x = np.zeros((phi_mat.shape[0]+1, phi_mat.shape[1]))
    q_y = np.zeros((phi_mat.shape[0], phi_mat.shape[1]+1))
    # derive each q_x - don't update boundary cells: keep zero flux
    for i in range(0, q_x.shape[0]-1):
        for j in range(1, q_x.shape[1]):
            rho_mat[i,j]
            t1 = (rho_mat[i, j-1] + rho_mat[i,j])/2
            t2 = Kappa((phi_mat[i,j-1]+phi_mat[i,j])/2)
            t3 = (rho_mat[i,j]-rho_mat[i,j-1])/dx
            q_x[i,j] = t1 * t2 * t3

    # derive each q_y
    for i in range(1, q_y.shape[0]):
        for j in range(0, q_y.shape[1]-1):
            q_y[i,j] = (rho_mat[i-1, j]+rho_mat[i,j])/2 * Kappa((phi_mat[i-1, j]+phi_mat[i-1, j])/2) * (rho_mat[i,j] - rho_mat[i-1,j])/dy

    # update q_x and q_y based on applied forcing
    q_x = q_x + add_qx_mat
    q_y = q_y + add_qy_mat

    # calculate total flux into each grid cell
    tot_flux = np.zeros(rho_mat.shape)
    for i in range(tot_flux.shape[0]-1):
        for j in range(tot_flux.shape[1]-1):
            x_comp = (q_x[i,j+1]-q_x[i,j])/dx
            y_comp = (q_y[i+1,j]-q_y[i,j])/dy
            if (x_comp !=0) or (y_comp !=0):
                print(i,j, x_comp, y_comp)
            tot_flux[i,j] = -(x_comp + y_comp)
    
    # update rho with the new flux
    rho_new = rho_mat+ tot_flux

    return rho_new

plt.imshow(update_rho(phi0, rho0, Kappa, additive_q, np.zeros((phi0.shape[0], phi0.shape[1]+1))))
plt.show()

