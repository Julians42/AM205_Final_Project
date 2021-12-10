from utils import *

# Constants
t_step = 0.05


# generate initial conditions
phi0 = np.genfromtxt("data/initial_conditions.txt")

vfluid0 = np.zeros(phi0.shape)
fdensity0 = np.zeros(phi0.shape)
press0 = np.zeros((phi0.shape[0]+1, phi0.shape[1]+1))

fluid_forcing(phi0.shape, t_step*1, 1)
