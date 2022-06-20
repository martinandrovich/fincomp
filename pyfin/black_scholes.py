import numpy as np
from math import log, sqrt, exp
import scipy.stats as st
from pyfin import gbm
import pyfin
from pyfin.pricing import feynman_kac


def bs(option_type, K, T, s0, r, sigma, t=0):
	"""Pricing of European options using analytical Black-Scholes."""

	d1 = (log(s0 / K) + (r + 1 / 2 * sigma ** 2) * (T - t)) / (sigma * sqrt(T - t))
	d2 = d1 - sigma * sqrt(T - t)

	if option_type == "CALL":
		V = s0 * st.norm.cdf(d1) - K * exp(-r * (T - t)) * st.norm.cdf(d2)
	else:  # OptionType.PUT
		V = K * exp(-r * (T - t)) * st.norm.cdf(-d2) - s0 * st.norm.cdf(-d1)

	return V


def bs_mc(option_type, K, T, s0, r, sigma, t=0, num_paths=1000, reproducible=False):
	"""Pricing of European options using simulated Black-Scholes with Monte Carlo/Feynman Kac."""

	# optimal version; compute only the last timeslice
	# based on (2.32)
	# S(t) = s0 * exp[(μ - 1/2 * σ²) * (T - t0) + σ * (W(t) - W(t0))]

	if reproducible:
		np.random.seed(pyfin.sde.DEFAULT_SEED)

	S = np.array([s0 * exp((r - 1/2 * sigma ** 2) * (T - t) + sigma * (np.random.normal(0, T) - np.random.normal(0, 1.0))) for _ in range(num_paths)])
	V = pyfin.pricing.feynman_kac(option_type, S, K, r, tau=T - t)

	return V


def bs_mc2(option_type, K, T, s0, r, sigma, t=0, num_paths=1000, reproducible=False):
	"""Pricing of European options using simulated Black-Scholes with Monte Carlo/Feynman Kac (slower)."""

	_, S, _ = gbm(s0=s0, mu=r, sigma=sigma, T=T, num_paths=num_paths, reproducible=reproducible)

	# payoff function lambda (for numpy)
	h = np.vectorize(lambda S: max(S - K, 0) if option_type == "CALL" else max(K - S, 0))

	H = h(S[:, -1])  # apply h(S) to each element
	V = exp(-r * (T - t)) * np.mean(H)

	return V


def bs_imp(option_type, market_price, s0, K, r, sigma_init, T, t=0):
	
	# Vega = dV/dsigma
	def vega(S_0, K, sigma, tau, r):
		d2   = (np.log(S_0 / float(K)) + (r - 0.5 * np.power(sigma,2.0)) * tau) / float(sigma * np.sqrt(tau))
		value = K * np.exp(-r * tau) * st.norm.pdf(d2) * np.sqrt(tau)
		return value
	
	error = 1e10
	sigma = sigma_init

	while error > 1e-10:
		e = market_price - bs(option_type, s0, K, r, sigma_init, T, t)
		e_prime = -vega(sigma)
		sigma_new = sigma - e/e_prime
		error = abs(sigma_new - sigma)
		sigma = sigma_new

	return sigma
