import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

from pyfin.sde import gbm_corr

mu = np.array([0.05, 0.1])
S = np.eye(2, 2)
C = np.eye(2, 2)

rho = -0.7
C = np.array([[1, rho], [rho, 1]])

t, Xs = gbm_corr(s0=1, mu=mu, S=S, C=C, T=1)
print(Xs)

plt.figure()
plt.plot(t, Xs[0, :])
plt.plot(t, Xs[1, :])
plt.grid()
plt.show()