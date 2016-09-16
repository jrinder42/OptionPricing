import numpy as np
import math
from random import gauss
from scipy.stats import gmean

# Exotic Options - Only European
class Asian:
	# Both Geometric and Arithmetic

	def __init__(self, S, X, r, sigma, T, steps, d = 0):
		self.S = float(S)
		self.X = X
		self.r = r
		self.sigma = sigma
		self.T = float(T)
		self.d = d
		self.steps = steps

	# option-put/call, type-geometric/arithmetic
	def Naive_MonteCarlo(self, option, type):
		simulations = 1000
		disc_factor = np.exp(-self.r * self.T)
		dt = self.T / self.steps
		payoffs = []
		if option.lower() == "call" and type.lower() == "arithmetic":
			for j in xrange(0, simulations):
				S_t = self.S
				sum = 0; average = 0
				for i in xrange(0, int(self.steps)):
					S_t = S_t * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * gauss(0,1.0))
					sum += S_t
				average = sum / self.steps
				payoffs.append(max(0.0, average - self.X))
			return disc_factor * np.sum(payoffs) / float(simulations)
		elif option.lower() == "call" and type.lower() == "geometric":
			for j in xrange(0, simulations):
				S_t = self.S
				average = 0; stock_hold = []
				for i in xrange(0, int(self.steps)):
					S_t = S_t * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * gauss(0,1.0))
					stock_hold.append(S_t)
				average = gmean(stock_hold)
				payoffs.append(max(0.0, average - self.X))
			return disc_factor * np.sum(payoffs) / float(simulations)
		elif option.lower() == "put" and type.lower() == "arithmetic":
			for j in xrange(0, simulations):
				S_t = self.S
				sum = 0; average = 0
				for i in xrange(0, int(self.steps)):
					S_t = S_t * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * gauss(0,1.0))
					sum += S_t
				average = sum / self.steps
				payoffs.append(max(0.0, self.X - average))
			return disc_factor * np.sum(payoffs) / float(simulations)
		elif option.lower() == "put" and type.lower() == "geometric":
			for j in xrange(0, simulations):
				S_t = self.S
				average = 0; stock_hold = []
				for i in xrange(0, int(self.steps)):
					S_t = S_t * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * gauss(0,1.0))
					stock_hold.append(S_t)
				average = gmean(stock_hold)
				payoffs.append(max(0.0, self.X - average))
			return disc_factor * np.sum(payoffs) / float(simulations)


	# Control Variate Monte Carlo
	def CV_MonteCarlo(self, option, type):
		pass
