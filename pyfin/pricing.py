import numpy as np
from math import exp


def feynman_kac(option_type, S_T, K, r, tau):

	# compute numerical price based on a vector of paths S(T)
	# tau = T - t0

	h = np.vectorize(lambda S: max(S - K, 0) if option_type == "CALL" else max(K - S, 0))
	H = h(S_T)  # apply h(S) to each element
	V = exp(-r * tau) * np.mean(H)

	return V
