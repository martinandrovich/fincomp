import numpy as np
from math import log, sqrt, exp, cos, sin, factorial, pi
import scipy.stats as st


def merton(option_type, K, T, s0, r, sigma, mu_J, sigma_J, xi_p, N=20, t0=0):
	"""Pricing of European options using analytical Merton."""

	# value of European option based on Merton model using analytical solution
	# based on (5.28)

	x0 = log(s0)
	tau = T - t0
	E_eJ = exp(mu_J + (sigma_J**2) / 2) - 1  # E[e^J - 1]

	def mu_hat(n):
		return x0 + (r - xi_p * E_eJ - 1/2 * sigma**2) * tau + n * mu_J

	def sigma_hat(n):
		return sqrt(sigma**2 + (n * sigma_J**2) / tau)

	def d1(n):
		return (log(s0 / K) + (r - xi_p * E_eJ - 1/2 * sigma**2 + sigma_hat(n)**2) * tau + n * mu_J) / (sigma_hat(n) * sqrt(tau))

	def d2(n):
		return d1(n) - sigma_hat(n) * sqrt(tau)

	def V_bar(n):
		return exp(mu_hat(n) + 1/2 * sigma_hat(n)**2 * tau) * st.norm.cdf(d1(n)) - K * st.norm.cdf(d2(n))

	def V(n):
		expected = sum([ ( (xi_p * tau)**k * exp(-xi_p * tau)) / factorial(k) * V_bar(k) for k in range(n)])
		return exp(-r * tau) * expected

	if option_type == "CALL":
		return V(N)
	else:
		return V(N) - s0 + K * exp(-r * tau)


def merton_cos(option_type, K, T, s0, r, sigma, mu_J, sigma_J, xi_p, a, b, N=1000, t0=0):
	"""Pricing of European options using COS-based Merton."""

	# value of European option based on Merton model using COS method
	# based on (6.28)

	i = complex(0, 1)
	tau = T - t0
	x0 = log(s0 / K)
	chf = merton_chf(tau, r, sigma, mu_J, sigma_J, xi_p, x0)
	H_k = merton_H_k(option_type, a, b, K)  # payout coefficients

	summand = lambda k: np.real(chf(k * pi/(b - a)) * np.exp(-i * k * pi * a/(b - a))) * H_k(k)
	V = exp(-r * tau) * sum([summand(k) * (1 if k else 0.5) for k in range(N)])  # multiply 1st term with 1/2

	return V


def merton_chf(t, r, sigma, mu_J, sigma_J, xi_p, x0=0):

	# charachteristic function of the Merton model
	# based on (5.33)

	def chf(u):

		i = complex(0, 1)

		E_eJ = np.exp(mu_J + (sigma_J**2) / 2) - 1  # E[e^J - 1]
		E_eiuJ = np.exp(i * u * mu_J - 1/2 * u**2 * sigma_J**2) - 1  # E[e^iuJ - 1]
		mu = r - 1/2 * sigma**2 - xi_p * E_eJ

		return np.exp(i * u * (x0 + mu * t) - 1/2 * sigma**2 * u**2 * t) * np.exp(xi_p * t * E_eiuJ)

	return chf


def merton_H_k(option_type, a, b, K):

	# cosine series coefficients of the payoff function V(T, y)
	# based on (6.34), (6.35)

	chi = lambda c, d, k: 1 / (1 + (k * pi/(b - a))**2) * (
		cos(k * pi * (d - a) / (b - a)) *
		exp(d) - cos(k * pi * (c - a) / (b - a)) * exp(c) + (k * pi) / (b - a) *
		sin(k * pi * (d - a) / (b - a)) * exp(d) - (k * pi) / (b - a) *
		sin(k * pi * (c - a) / (b - a)) * exp(c) )

	psi = lambda c, d, k: (sin(k * pi * (d - a) / (b - a)) - sin(k * pi * (c - a) / (b - a))) * \
		(b - a) / (k * pi) if k else (d - c)

	H_kc = lambda k: 2/(b - a) * K * (chi(0, b, k) - psi(0, b, k))
	H_kp = lambda k: 2/(b - a) * K * (-chi(a, 0, k) + psi(a, 0, k))

	return H_kc if option_type == "CALL" else H_kp
