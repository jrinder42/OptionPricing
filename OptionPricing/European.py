import numpy as np
from random import gauss
from scipy.stats import norm



class European:

	'''
	S: current stock price
	X: strike price
	r: risk-free rate
	sigma: volatility
	T: Time until maturity
	d: dividend yield
	'''

	def __init__(self, S, X, r, sigma, T, d = 0):
		self.X = X
		self.S = float(S)
		self.r = r
		self.sigma = sigma
		self.T = T
		self.d = d
		self.b = self.r - self.d

	def MonteCarlo(self, option):
		if option.lower() == "call":
			simulations = 50000
			disc_factor = np.exp(-self.r * self.T)
			payoffs = []
			for i in xrange(simulations):
				S_t = self.S * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * gauss(0,1.0))
				payoffs.append(max(0.0, S_t - self.X))
			return disc_factor * (sum(payoffs) / float(simulations))
		elif option.lower() == "put":
			simulations = 50000
			disc_factor = np.exp(-self.r * self.T)
			payoffs = []
			for i in xrange(simulations):
				S_t = self.S * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * gauss(0,1.0))
				payoffs.append(max(0.0,self.X - S_t))
			return disc_factor * (sum(payoffs) / float(simulations))
		else:
			return "Invalid data"


	def BlackScholes(self, option):
		d1 = (np.log(self.S/self.X) + ((self.r-self.d) + ((self.sigma**2)*0.5))*self.T) / (self.sigma*np.sqrt(self.T))
		d2 = d1 - self.sigma*np.sqrt(self.T)
		if option.lower() == "call":
			return self.S * np.exp(-self.d*self.T) * norm.cdf(d1) - self.X * np.exp(-self.r*self.T) * norm.cdf(d2)
		elif option.lower() == "put":
			return self.X * np.exp(-self.r*self.T) * norm.cdf(-d2) - self.S * np.exp(-self.d*self.T) * norm.cdf(-d1)
		else:
			return "Invalid data"

	# Cox-Ross-Rubenstein Binomial Pricing
	def CRR(self, option, steps):
		self.T = float(self.T)
		dt = round(self.T/steps, 5)
		v = self.r - self.d
		up = np.exp(v * np.sqrt(dt))
		down = 1./up
		p = (np.exp(self.r * dt) - down)/(up - down)


		# Binomial Price Tree
		val = np.zeros((steps + 1, steps + 1))
		val[0, 0] = self.S
		for i in xrange(1, steps + 1):
			val[i, 0] = val[i - 1, 0] * up
			for j in xrange(1, i + 1):
				val[i, j] = val[i - 1, j - 1] * down

		# Option value at each node
		price = np.zeros((steps + 1, steps + 1))
		for i in xrange(steps + 1):
			if option.lower() == "call":
				price[steps, i] = max(0, val[steps, i] - self.X)
			elif option.lower() == "put":
				price[steps, i] == max(0, self.X - val[steps, i])

		# Backward recursion for option price
		for i in xrange(steps - 1, -1, -1):
			for j in xrange(i + 1):
				price[i, j] = np.exp(-self.r*dt)*(p*price[i+1,j] + (1-p)*price[i+1,j+1])

		return price[0, 0]


