import numpy as np
from math import sqrt

DEFAULT_SEED = 0

def abm(s0, mu, sigma, T, dt=0.01, num_paths=10, reproducible=False):
	"""Arithmeic brownian motion."""

	# dS(t) = mu * dt + sigma * dW
	# S(t + dt) = S(t) + mu * dt + sigma * (W(t + dt) - W(t))
	# S(t + dt) = S(t) + dS(t) * dt

	# compute multiple paths at once
	# each path/realization is a row with column-wise time progression

	if reproducible:
		np.random.seed(DEFAULT_SEED)

	num_steps = int(T / dt)
	Z = np.random.normal(0.0, 1.0, [num_paths, num_steps])
	W = np.zeros([num_paths, num_steps + 1])
	S = np.zeros([num_paths, num_steps + 1])

	t = np.linspace(0, T, num_steps + 1)

	S[:, 0] = s0

	for i in range(num_steps):
		if num_paths > 1:  # standardize samples (mean: 0, var: 1)
			Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])

		# wiener process
		W[:, i + 1] = W[:, i] + dt ** 0.5 * Z[:, i]
		dW = W[:, i + 1] - W[:, i]

		# stock dynamics
		S[:, i + 1] = S[:, i] + mu * dt + sigma * dW
		# S[:, i + 1] = S[:, i] + (mu * dt + sigma * dW) * dt

		dW = Z[:, i] * sqrt(dt)
		S[:, i + 1] = S[:, i] + mu * dt + sigma * dW

	return t, S


def gbm(s0, mu, sigma, T, dt=0.01, num_paths=10, reproducible=False):
	"""Geometric brownian motion."""

	# dS(t) = mu * S(t)*dt + sigma * S(t)*dW(t)
	# S(t + dt) = S(t) + mu * S(t)*dt + sigma * S(t) * (W(t + dt) - W(t))
	# S(t) = S0 * exp((mu - sigma^2/2)*t + sigma * W(t))

	if reproducible:
		np.random.seed(DEFAULT_SEED)

	num_steps = int(T / dt)
	Z = np.random.normal(0.0, 1.0, [num_paths, num_steps])
	W = np.zeros([num_paths, num_steps + 1])
	X = np.zeros([num_paths, num_steps + 1])
	S = np.zeros([num_paths, num_steps + 1])

	t = np.linspace(0, T, num_steps + 1)

	S[:, 0] = s0
	X[:, 0] = np.log(s0)

	for i in range(0, num_steps):

		if num_paths > 1:  # standardize samples (mean: 0, var: 1)
			Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])

		# wiener process
		W[:, i + 1] = W[:, i] + dt ** 0.5 * Z[:, i]
		dW = W[:, i + 1] - W[:, i]

		# stock dynamics
		dW = Z[:, i] * sqrt(dt)  # using dW ~ N(0, dt) = N(0, 1) * sqrt(dt)
		S[:, i + 1] = S[:, i] + mu * S[:, i] * dt + sigma * S[:, i] * dW

		# under log-transform (same result)
		X[:, i + 1] = X[:, i] + (mu - 1/2 * sigma ** 2) * dt + sigma * dW
		# S[:, i + 1] = np.exp(X[:, i + 1])

	return t, S, X


def ou(s0, kappa, theta, sigma, T, dt=0.01, num_paths=10, reproducible=False):
	"""Geometric brownian motion."""

	# dS(t) = κ * (θ - S(t)) * dt + σ * dW
	# S(t + dt) = S(t) + κ * (θ - S(t)) * dt + σ * dW

	if reproducible:
		np.random.seed(DEFAULT_SEED)

	num_steps = int(T / dt)
	Z = np.random.normal(0.0, 1.0, [num_paths, num_steps])
	W = np.zeros([num_paths, num_steps + 1])
	S = np.zeros([num_paths, num_steps + 1])
	t = np.linspace(0, T, num_steps + 1)

	S[:, 0] = s0

	for i in range(0, num_steps):

		if num_paths > 1:  # standardize samples (mean: 0, var: 1)
			Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])

		# wiener process
		W[:, i + 1] = W[:, i] + dt ** 0.5 * Z[:, i]
		dW = W[:, i + 1] - W[:, i]

		# stock dynamics
		dW = Z[:, i] * sqrt(dt)  # using dW ~ N(0, dt) = N(0, 1) * sqrt(dt)
		S[:, i + 1] = S[:, i] + kappa * (theta - S[:, i]) * dt + sigma * dW

	return t, S

def abm_corr(s0, mu, C, D, T, dt=0.01, reproducible=False):

	if reproducible:
		np.random.seed(DEFAULT_SEED)

	num_steps = int(T / dt)
	num_paths = len(mu)

	Z = np.random.normal(0.0, 1.0, [num_paths, num_steps])
	S = np.zeros([num_paths, num_steps + 1])
	t = np.linspace(0, T, num_steps + 1)
	L = np.linalg.cholesky(C)
	sigma = D @ L

	for i in range(0, num_steps):

		dW = Z[:, i] * sqrt(dt)
		S[:, i + 1] = S[:, i] + mu * dt + sigma @ dW

	return t, S
