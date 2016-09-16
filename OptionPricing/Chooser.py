import numpy as np
from scipy.stats import norm


class Chooser:

    def __init__(self, S, X, r, sigma, t1, T, d = 0):
        self.S = S
        self.X = X
        self.r = r
        self.sigma = sigma
        self.t1 = t1
        self.T = T
        self.d = d

    def BlackScholes(self):
        # For call
        d1_C = (self.r - self.d + 0.5*(self.sigma**2)*self.T) / (self.sigma * np.sqrt(self.T))
        d2_C = d1_C - self.sigma*np.sqrt(self.T)
        # For put
        d1_P = (np.log(self.S/(self.X*np.exp(-self.d))) + (self.r-self.d+0.5*(self.sigma**2))*self.t1) / (self.sigma*np.sqrt(self.t1))
        d2_P = d1_P - self.sigma*np.sqrt(self.t1)

        Euro_C = self.S*np.exp(-self.d*self.T)*norm.cdf(d1_C) - self.X*np.exp(-self.r*self.T)*norm.cdf(d2_C)
        Euro_P = self.X*np.exp(-self.d)*np.exp(-self.r*self.t1)*norm.cdf(-d2_P) - self.S*np.exp(-self.d*self.t1)*norm.cdf(-d1_P)
        return Euro_C + np.exp(-self.d*(self.T-self.t1)) * Euro_P
