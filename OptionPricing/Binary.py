import numpy as np
from scipy.stats import norm


class Binary:

    def __init__(self, S, X, r, sigma, T, d = 0):
        self.S = float(S)
        self.X = X
        self.r = r
        self.sigma = sigma
        self.T = float(T)
        self.d = d

    def BlackScholes(self, option, type):
        d1 = ((self.S/self.X) + (self.r-self.d + (self.sigma**2)/2)) / (self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)

        # Cash or Nothing
        if option.lower() == "call" and type.lower() == "con":
            return np.exp(-self.r*self.T) * norm.cdf(d2)
        elif option.lower() == "put" and type.lower() == "con":
            return np.exp(-self.r*self.T) * norm.cdf(-d2)

        # Asset or Nothing
        elif option.lower() == "call" and type.lower() == "aon":
            return self.S * np.exp(-self.d*self.T) * norm.cdf(d1)
        elif option.lower() == "put" and type.lower() == "aon":
            return self.S * np.exp(-self.d*self.T) * norm.cdf(-d1)
        else:
            "Invalid data"
