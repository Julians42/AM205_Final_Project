import numpy as np
from math import tanh
import matplotlib.pyplot as plt
xs = np.linspace(0, 1, 1000)

def psi(phi, c1 =0.1, w=5, phis=0.6, c2=0):
    return c1*tanh(w*(phi-phis))+c2



psiv = np.vectorize(psi)
ys = psiv(xs)#[psi(x) for x in xs]

plt.plot(xs, ys)
plt.show()