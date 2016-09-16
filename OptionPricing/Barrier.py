import numpy as np

from scipy.stats import norm

from European import European


# Change so don't have to imput volatility sigma
class Barrier:
	# H is barrier level
	def __init__(self, S, X, r, sigma, T, H, d = 0):
		self.S = float(S)
		self.X = X
		self.r = r
		self.sigma = sigma
		self.T = T
		self.H = H
		self.d = d

	def BlackScholes(self, type):
		v = self.r - self.d - (self.sigma ** 2) / 2
		obj1 = European((self.H ** 2) / self.S, self.X, self.r, self.T, self.d)
		obj2 = European((self.H ** 2) / self.S, self.H, self.r, self.T, self.d)
		obj3 = European(self.S, self.H, self.r, self.T, self.d)
		obj4 = European(self.S, self.X, self.r, self.T, self.d)
		d_bs1 = (np.log(self.H/self.S) + v * self.T) / (self.sigma * self.T)
		d_bs2 = (np.log(self.S/self.H) + v * self.T) / (self.sigma * self.T)

		# Up and In
		if type.lower() == "ui":
			return ((self.H/self.X) ** (2*v / (self.sigma ** 2))) \
				* (obj1.BlackScholes("put") - obj2.BlackScholes("put") + (self.H - self.X) * np.exp(-self.r*self.T) \
				* norm.cdf(-d_bs1, loc = 0, scale = 1)) + obj3.BlackScholes("call") + (self.H - self.X) * np.exp(-self.r*self.T) \
				* norm.cdf(d_bs2)
		# Up and Out
		elif type.lower() == "uo":
			return obj4.BlackScholes("call") - obj3.BlackScholes("call") - (self.H - self.X) * np.exp(-self.r*self.T) \
				- (self.H/self.S)**(2*v/(self.sigma**2)) * (obj1.BlackScholes("call") - obj2.BlackScholes("call") \
				- (self.H - self.X)*np.exp(-self.r*self.T)*norm.cdf(d_bs1))
		# Down and In
		elif type.lower() == "di":
			return (self.H/self.S) ** (2*v/(self.sigma**2)) * obj1.BlackScholes("call")
		# Down and Out
		elif type.lower() == "do":
			return obj4.BlackScholes("call") - (self.H/self.S) ** (2*v/(self.sigma**2)) * obj1.BlackScholes("call")

	def Binomial(self):
		pass