import numpy as np

from scipy.stats import norm

class Exchange:

    '''
    S: current stock price
    X: strike price of the underlying option (amount paid to purchase the stock)
    r: risk-free rate
    sigma_S: volatility of the underlying asset
    sigma_Q: volatility of the exchange asset
    p: correlation between both assets
    T: expiry of the underlying option
    d: continuous dividend rate on the stock
    '''

    def __init__(self, S, Q, sigma_S, sigma_Q, p, T, d_S = 0, d_Q = 0):
        self.S = S
        self.Q = Q
        self.sigma_S = sigma_S
        self.sigma_Q = sigma_Q
        self.p = p
        self.T = T
        self.d_S = d_S
        self.d_Q = d_Q

    def BlackScholes(self, option):
        sigma = np.sqrt(self.sigma_S**2 + self.sigma_Q**2 - 2*self.p*self.sigma_S*self.sigma_Q)
        d1 = (np.log(self.S*np.exp(-self.d_S*self.T)/(self.Q*np.exp(-self.d_Q*self.T))) + 0.5*sigma**2 * self.T) \
             / (sigma * self.T)
        d2 = d1 - sigma*self.T
        if option.lower() == "call":
            return self.S*np.exp(-self.d_S*self.T)*norm.cdf(d1) - self.Q*np.exp(-self.d_Q*self.T)*norm.cdf(d2)
        elif option.lower() == "put":
            return self.Q*np.exp(-self.d_Q*self.T)*norm.cdf(-d2) - self.S*np.exp(-self.d_S*self.T)*norm.cdf(-d1)