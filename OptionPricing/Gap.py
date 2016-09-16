import numpy as np

from scipy.stats import norm

class Gap:

    def __init__(self, S, K1, K2, r, sigma, T, d = 0):
        '''
	S: current stock price
	K1: strike price
	K2: trigger price
	r: risk-free rate
	sigma: volatility
	T: Time until maturity
	d: dividend yield
	'''

        self.S = float(S)
        self.K1 = K1
        self.K2 = K2
        self.r = r
        self.sigma = sigma
        self.T = T
        self.d = d

    def BlackScholes(self, option):
        d1 = ((np.log(self.S*np.exp(-self.d*self.T) / (self.K2*np.exp(-self.r*self.T))) + (self.sigma**2)/2*self.T)) \
                / self.sigma * np.sqrt(self.T)
        d2 = d1 - self.sigma*np.sqrt(self.T)
        if option.lower() == "call":
            return self.S * np.exp(-self.d*self.T) * norm.cdf(d1) - self.K1 * np.exp(-self.r*self.T) * norm.cdf(d2)
        elif option.lower() == "put":
            return self.K1 * np.exp(-self.r*self.T) * norm.cdf(-d2) - self.S * np.exp(-self.d*self.T) * norm.cdf(-d1)
        else:
            return "Invalid data"