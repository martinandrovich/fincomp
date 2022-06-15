import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

np.random.seed(2)

s0 = 1
mu = 0.05
sigma = 0.7
T = 1
num_steps = 1000
dt = T / float(num_steps)

Z = np.random.normal(0.0, 1.0, num_steps)
W = np.zeros(num_steps + 1)
X_ABM = np.zeros(num_steps + 1)
X_GBM = np.zeros(num_steps + 1)
X_OU = np.zeros(num_steps + 1)
t = np.zeros(num_steps + 1)

X_ABM[0], X_GBM[0], X_OU[0] = s0, s0, s0

for i in range(num_steps):
	W[i + 1] = W[i] + dt ** 0.5 * Z[i]
	dW = W[i + 1] - W[i]
	# dW = Z[i] * sqrt(dt)

	X_ABM[i + 1] = X_ABM[i] + mu * dt + sigma * dW
	X_GBM[i + 1] = X_GBM[i] + mu * X_GBM[i] * dt + sigma * X_GBM[i] * dW
	X_OU[i + 1] = X_OU[i] + dt * (mu * dt + sigma * dW)

	t[i + 1] = t[i] + dt

fig, axs = plt.subplots(1, 3, figsize=(10, 5))

# plt.plot(t, X_ABM.)

axs[0].plot(t, X_ABM.T)
axs[0].set_title("Arithmetic Brownian Motion (ABM)")

axs[1].plot(t, X_GBM.T)
axs[1].set_title("Geometric Brownian Motion (GBM)")

axs[2].plot(t, X_OU.T)
axs[2].set_title("Arithmetic Brownian Motion (ABM)")

plt.show()

