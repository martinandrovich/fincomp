import numpy as np
from math import log, sqrt, exp
import scipy.stats as st
from pyfin import gbm


class OptionType:
	PUT = "PUT"
	CALL = "CALL"


def bs(option_type, s0, K, r, sigma, T, t=0):

	d1 = (log(s0 / K) + (r + 1 / 2 * sigma ** 2) * (T - t)) / (sigma * sqrt(T - t))
	d2 = d1 - sigma * sqrt(T - t)

	if option_type == OptionType.CALL:
		V = s0 * st.norm.cdf(d1) - K * exp(-r * (T - t)) * st.norm.cdf(d2)
	else:  # OptionType.PUT
		V = K * exp(-r * (T - t)) * st.norm.cdf(-d2) - s0 * st.norm.cdf(-d1)

	return V


def bs_num(option_type, s0, K, r, sigma, T, t=0, num_paths=1000):

	# optimal version; compute only the last timeslice
	# S(t) = s0 * exp[(μ - 1/2 * σ²) * (t - t0) + σ * (W(t) - W(t0))]

	S = np.array([s0 * exp((r - 1/2 * sigma ** 2) * (T - t) + sigma * (np.random.normal(0, T) - np.random.normal(0, 1.0))) for _ in range(num_paths)])

	# payoff function lambda (for numpy)
	h = np.vectorize(lambda S: max(S - K, 0) if option_type == OptionType.CALL else max(K - S, 0))

	H = h(S)  # apply h(S) to each element
	V = exp(-r * (T - t)) * np.mean(H)

	return V


def bs_num2(option_type, s0, K, r, sigma, T, t=0, num_paths=1000):

	_, S, _ = gbm(s0=s0, mu=r, sigma=sigma, T=T, num_paths=num_paths)

	# payoff function lambda (for numpy)
	h = np.vectorize(lambda S: max(S - K, 0) if option_type == OptionType.CALL else max(K - S, 0))

	H = h(S[:, -1])  # apply h(S) to each element
	V = exp(-r * (T - t)) * np.mean(H)

	return V
