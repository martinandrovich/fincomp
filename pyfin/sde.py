import numpy as np
from math import sqrt, exp, log

DEFAULT_SEED = 0


def abm(s0, mu, sigma, T, dt=0.01, num_paths=10, reproducible=False):
	"""Arithmeic Brownian motion."""

	# based on (2.1), (2.32)
	# dS(t) = mu * dt + sigma * dW
	# S(t + dt) = S(t) + mu * dt + sigma * (W(t + dt) - W(t))

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
	"""Geometric Brownian motion."""

	# based on (2.1)
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

	for i in range(num_steps):

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
	"""Ornstein-Uhlenbeck."""

	# based on (8.1)
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

	for i in range(num_steps):

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
	"""
	Correlated arithmetic Brownian motion.
	:param np.array() D: design matrix
	:param np.array() C: correlation matrix
	"""

	# based on (7.8)

	if reproducible:
		np.random.seed(DEFAULT_SEED)

	num_steps = int(T / dt)
	num_paths = len(mu)

	Z = np.random.normal(0.0, 1.0, [num_paths, num_steps])
	S = np.zeros([num_paths, num_steps + 1])
	t = np.linspace(0, T, num_steps + 1)
	L = np.linalg.cholesky(C)
	sigma = D @ L

	S[:, 0] = s0

	for i in range(0, num_steps):

		dW = Z[:, i] * sqrt(dt)
		S[:, i + 1] = S[:, i] + mu * dt + sigma @ dW

	return t, S


def merton(s0, r, sigma, mu_J, sigma_J, xi_p, T, dt=0.01, num_paths=10, reproducible=False):
	"""Standard jump diffusion model (Merton model)."""

	# based on (5.10), (5.11)
	# dX(t) = (r - xi_p * E[e^J - 1] - 1/2 * sigma^2)*dt + sigma*dW(t) * J*dXp(t)

	if reproducible:
		np.random.seed(DEFAULT_SEED)

	num_steps = int(T / dt)

	P = np.random.poisson(xi_p * dt, [num_paths, num_steps + 1])
	Z = np.random.normal(0.0, 1.0, [num_paths, num_steps])
	J = np.random.normal(mu_J, sigma_J, [num_paths, num_steps])
	X = np.zeros([num_paths, num_steps + 1])
	t = np.linspace(0, T, num_steps + 1)

	X[:, 0] = log(s0)

	for i in range(0, num_steps):

		if num_paths > 1:  # standardize samples (mean: 0, var: 1)
			Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])

		w = xi_p * (exp(mu_J + (sigma_J ** 2) / 2) - 1) + 1/2 * sigma**2  # drift correction term
		dW = sqrt(dt) * Z[:, i]
		# dP = P[:, i] - P[:, 0]
		dP = P[:, i]
		X[:, i + 1] = X[:, i] + (r - w) * dt + sigma * dW + J[:, i] * dP

	S = np.exp(X)

	return t, S, X


def heston(s0, r, v0, v_bar, kappa, gamma, rho, T, dt=0.01, num_paths=10,  reproducible=False):
	"""Heston stochastic volatility model."""

	# based on (8.18), (8.21), (8.3)
	# dS(t) = r * S(t)dt + sqrt(v(t)) * S(t) * dW_x(t)
	# dv(t) = κ(v_bar - v(t))*dt + γ*sqrt(v(t)) * [ρ*dW_x(t) + sqrt(1 - ρ^2)*dW_v(t)]
	# dW_v*dW_x = ρ*dt

	if reproducible:
		np.random.seed(DEFAULT_SEED)

	num_steps = int(T / dt)
	Z_x = np.random.normal(0.0, 1.0, [num_paths, num_steps])
	Z_v = np.random.normal(0.0, 1.0, [num_paths, num_steps])
	S = np.zeros([num_paths, num_steps + 1])
	V = np.zeros([num_paths, num_steps + 1])
	t = np.linspace(0, T, num_steps + 1)

	S[:, 0] = s0
	V[:, 0] = v0

	for i in range(num_steps):

		if num_paths > 1:  # standardize samples (mean: 0, var: 1)
			Z_x[:, i] = (Z_x[:, i] - np.mean(Z_x[:, i])) / np.std(Z_x[:, i])
			Z_v[:, i] = (Z_v[:, i] - np.mean(Z_v[:, i])) / np.std(Z_v[:, i])

		dW_x, dW_v = sqrt(dt) * Z_x[:, i], sqrt(dt) * Z_v[:, i]

		S[:, i + 1] = S[:, i] + r * S[:, i] * dt + np.sqrt(V[:, i]) * S[:, i] * dW_x
		V[:, i + 1] = V[:, i] + kappa * (v_bar - V[:, i]) * dt + gamma * np.sqrt(V[:, i]) * (rho * dW_x + sqrt(1 - rho**2) * dW_v)
		V[:, i + 1] = np.maximum(V[:, i + 1], 0)  # truncate due to sqrt(v(t))

	X = np.log(S)

	return t, S, X, V


def heston_corr(s0, r, v0, v_bar, kappa, gamma, rho, T, dt=0.01, num_paths=10, reproducible=False):

	# based on (8.38)

	if reproducible:
		np.random.seed(DEFAULT_SEED)

	num_steps = int(T / dt)
	Z_x = np.random.normal(0.0, 1.0, [num_paths, num_steps])
	Z_v = np.random.normal(0.0, 1.0, [num_paths, num_steps])
	V = np.zeros([num_paths, num_steps + 1])
	X = np.zeros([num_paths, num_steps + 1])
	t = np.linspace(0, T, num_steps + 1)

	X[:, 0] = log(s0)
	V[:, 0] = v0

	for i in range(num_steps):

		if num_paths > 1:  # making sure that samples from normal have mean 0 and variance 1
			Z_x[:, i] = (Z_x[:, i] - np.mean(Z_x[:, i])) / np.std(Z_x[:, i])
			Z_v[:, i] = (Z_v[:, i] - np.mean(Z_v[:, i])) / np.std(Z_v[:, i])

		Z_v[:, i] = rho * Z_x[:, i] + np.sqrt(1 - rho**2) * Z_v[:, i]  # correlate the processes

		dW_x = sqrt(dt) * Z_x[:, i]
		dW_v = sqrt(dt) * Z_v[:, i]
		X[:, i + 1] = X[:, i] + (r - 1/2 * V[:, i]) * dt + np.sqrt(V[:, i]) * dW_x
		V[:, i + 1] = V[:, i] + kappa * (v_bar - V[:, i]) * dt + gamma * np.sqrt(V[:, i]) * dW_v

		# manual in case Z_x and Z_v are not correlated
		# X[:, i + 1] = X[:, i] + (r - 1/2 * V[:, i]) * dt + np.sqrt(V[:, i]) * dW_x
		# V[:, i + 1] = V[:, i] + kappa * (v_bar - V[:, i]) * dt + rho * gamma * np.sqrt(V[:, i]) * dW_x + gamma * np.sqrt((1 - rho**2) * V[:, i]) * dW_v

		V[:, i + 1] = np.maximum(V[:, i + 1], 0)  # truncate due to sqrt(v(t))

	S = np.exp(X)

	return t, S, X, V
