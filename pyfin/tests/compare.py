from merton import MertonCallPrice, OptionType
from pyfin.merton import merton
import numpy as np


s0 = 100
r = 0.05
sigma = 0.1
mu_J = 0
sigma_J = 0.5
xi_p = 1
T = 5
t0 = 0
N = 10
K = range(80, 100, 5)
option_type = OptionType.CALL

tau = T - t0


def compare_merton():
    lech_price = MertonCallPrice(option_type, s0, np.array(list(K)), r, tau, mu_J, sigma_J, sigma, xi_p)
    our_price = [merton(option_type=option_type, s0=s0, r=r, sigma=sigma, mu_J=mu_J, sigma_J=sigma_J, xi_p=xi_p, K=k, T=T, N=100) for k in K]
    # print(lech_price)
    [print(f"{lech_p = }, {our_p = }") for lech_p, our_p in zip(lech_price, our_price)]




if __name__ == "__main__":
    compare_merton()
