import pyfin
import numpy as np
from math import log, sqrt, exp, cos, sin, factorial, pi
import enum
import scipy.stats as st


def heston_mc(option_type, K, T, s0, r, v0, v_bar, kappa, gamma, rho, dt=0.01, num_paths=10, t0=0, reproducible=False):
	"""Pricing of European options using simulated Heston model with Monte Carlo/Feynman Kac."""

	_, S, _, _ = pyfin.sde.heston(s0, r, v0, v_bar, kappa, gamma, rho, T, dt, num_paths, reproducible)
	V = pyfin.pricing.feynman_kac(option_type, S[:, -1], K, r, tau=T-t0)

	return V


def heston_aes(option_type, K, T, s0, r, v0, v_bar, kappa, gamma, rho, dt=0.01, num_paths=10, t0=0, reproducible=False):
	"""Pricing of European options using almost-exact simulated Heston model with Monte Carlo/Feynman Kac."""

	_, S, _, _ = pyfin.sde.heston_aes(s0, r, v0, v_bar, kappa, gamma, rho, T, dt, num_paths, reproducible)
	V = pyfin.pricing.feynman_kac(option_type, S[:, -1], K, r, tau=T-t0)

	return V


def heston_cos(option_type, K, T, s0, r, v0, v_bar, kappa, gamma, rho, a, b, N=1000, t0=0):
	"""Pricing of European options using COS-based Heston."""

	i = complex(0, 1)
	tau = T - t0
	x0 = np.log(s0/K)
	chf = heston_chf(r, kappa, rho, v_bar, gamma, T, t0, v0, x0)
	H_k = heston_H_k(option_type, a, b, K)

	summand = lambda k: np.real(chf(k * pi/(b - a)) * np.exp(-i * k * pi * a/(b - a))) * H_k(k)
	V = exp(-r * tau) * sum([summand(k) * (1 if k else 0.5) for k in range(N)])  # multiply 1st term with 1/2
	return V


def heston_chf(r, kappa, rho, v_bar, gamma, T, t0, v0, x0):

	# charachteristic function of the Heston model
	# based on (8.54)

	def chf(u):
		tau = T - t0
		i = complex(0, 1)
		expr = kappa - i * rho * gamma * u
		D1 = np.sqrt(expr**2 + (u**2 + i * u)*gamma**2)
		g = (expr - D1)/(expr + D1)

		return np.exp(i * u * x0) * \
			np.exp(i * u * r * tau + v0/(gamma**2) * (1 - np.exp(-D1 * tau)) / (1 - g * np.exp(-D1 * tau)) * (expr - D1)) * \
			np.exp(kappa * v_bar/(gamma**2) * (tau * (expr - D1) - 2 * np.log((1 - g * np.exp(-D1 * tau)))/(1 - g)))

	return chf


def heston_H_k(option_type, a, b, K):

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
