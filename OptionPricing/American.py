import numpy as np
import math
from European import *
from scipy.stats import norm

class American:

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


	# Cox-Ross-Rubenstein Binomial Pricing
	def CRR(self, option, steps):
		self.T = float(self.T)
		dt = round(self.T/steps, 5)
		v = self.r - self.d
		up = np.exp(self.sigma * np.sqrt(dt))
		down = np.exp(-self.sigma * np.sqrt(dt))
		up = 1.2
		down = 0.8
		p = (np.exp(v * dt) - down)/(up - down)


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
				price[steps, i] = max(0, self.X - val[steps, i])


		# Backward recursion for option price
		for i in xrange(steps - 1, -1, -1):
			for j in xrange(i + 1):
				if option.lower() == "call":
					price[i, j] = max(val[i,j] - self.X, np.exp(-self.r*dt)*(p*price[i+1,j] + (1-p)*price[i+1,j+1]))
				elif option.lower() == "put":
					price[i, j] = max(self.X - val[i, j], np.exp(-self.r*dt)*(p*price[i+1,j] + (1-p)*price[i+1,j+1]))

		return price[0, 0]

	# Quadratic Approximation - Can't have negative volatility
	def QuadApproxCall(self):
		Accuracy = math.pow(10, -6)
		b = self.r - self.d

		sigma2 = self.sigma**2
		time_sqrt = math.sqrt(self.T)
		nn = 2.*self.b/sigma2
		m = 2.*self.r/sigma2
		k = 1. - math.exp(-self.r*self.T)
		q2 = (-(nn-1) + math.sqrt(math.pow((nn-1),2) + (4*m/k)))/2

		# seed value from the paper
		q2_inf = (-(nn-1) + math.sqrt(math.pow((nn-1),2) + 4*m))/2
		'''
		S_star_inf = self.X / (1. - 1./q2_inf)
		h2 = -(self.b*self.T+2.*self.sigma*time_sqrt)*(self.X/(S_star_inf-self.X))
		S_seed = self.X + (S_star_inf-self.X)*(1.0-math.exp(h2))

		no_iterations=0 # iterate on S to find S_star, using Newton steps
		Si=S_seed
		g=1.
		gprime=1.
		Eu = European(Si, self.X, self.r, self.sigma, self.T, self.d)
		while ((math.fabs(g) > Accuracy) and (math.fabs(gprime)>Accuracy) and ( no_iterations+1<500) and (Si>0.)):
			c = Eu.BlackScholes("Call")
			d1 = float((math.log(Si/self.X)+(b+0.5*sigma2)*self.T)/(self.sigma*time_sqrt))
			g=(1.-1./q2)*Si-self.X-c+(1./q2)*Si*math.exp((b-self.r)*self.T)*norm.cdf(d1)
			gprime=(1.-1./q2)*(1.-math.exp((b-self.r)*self.T)*norm.cdf(d1))+(1./q2)\
					*math.exp((b-self.r)*self.T)*norm.pdf(d1)*(1.0/(self.sigma*time_sqrt))
			Si=Si-(g/gprime)

		Eu = European(self.S, self.X, self.r, self.sigma, self.T, self.d)
		if (math.fabs(g)>Accuracy):
			S_star = float(S_seed) # did not converge
		else:
			S_star = Si
		C=0.
		c = Eu.BlackScholes("Call")
		if (self.S>=S_star):
			C=self.S-self.X
		else:
			d1 = float(math.log(S_star/self.X)+(b+0.5*sigma2)*self.T)/(self.sigma*time_sqrt)
			A2 =  (1.0-math.exp((b-self.r)*self.T)*norm.cdf(d1))* (S_star/q2)
			C=c+A2*math.pow((self.S/S_star),q2)

		return max(C,c) # know value will never be less than BS value
		'''
		return (-(nn-1) + math.sqrt(math.pow((nn-1),2) + 4*m))/2

	def QuadApproxPut(self):
		Accuracy = math.pow(10, -6)
		b = self.r - self.d

		sigma2 = self.sigma**2
		time_sqrt = math.sqrt(self.T)
		nn = 2.*self.b/sigma2
		m = 2.*self.r/sigma2
		k = 1. - math.exp(-self.r*self.T)
		q1 = (-(nn-1) - math.sqrt(math.pow((nn-1),2) + (4*m/k)))*0.5

		q1_inf = (-(nn-1) - math.sqrt(math.pow((nn-1),2) + 4*m))*0.5
		S_star2_inf = self.X/(1-1./q1_inf)
		h1 = (b*self.T - 2*self.sigma*math.sqrt(self.T)) * (self.X/(self.X - S_star2_inf))
		S_seed = S_star2_inf + (self.X - S_star2_inf)*math.exp(h1)

		no_iterations=0 # iterate on S to find S_star, using Newton steps
		Si=S_seed
		g=1.
		gprime=1.
		Eu = European(Si, self.X, self.r, self.sigma, self.T, self.d)
		while ((math.fabs(g) > Accuracy) and (math.fabs(gprime)>Accuracy) and ( no_iterations+1<500) and (Si>0.)):
			p = Eu.BlackScholes("put")
			d1 = float((math.log(Si/self.X)+(b+0.5*sigma2)*self.T)/(self.sigma*time_sqrt))
			#g=(1.-1./q1)*Si-self.X-c+(1./q1)*Si*math.exp((b-self.r)*self.T)*norm.cdf(d1)
			gprime=(1.-1./q1)*(1.-math.exp((b-self.r)*self.T)*norm.cdf(d1))+(1./q1)\
					*math.exp((b-self.r)*self.T)*norm.pdf(d1)*(1.0/(self.sigma*time_sqrt))
			Si=Si-(g/gprime)

		Eu = European(self.S, self.X, self.r, self.sigma, self.T, self.d)
		if (math.fabs(g)>Accuracy):
			S_star = float(S_seed) # did not converge
		else:
			S_star = Si
		C=0.
		c = Eu.BlackScholes("Call")
		if (self.S>=S_star):
			C=self.S-self.X
		else:
			d1 = float(math.log(S_star/self.X)+(b+0.5*sigma2)*self.T)/(self.sigma*time_sqrt)
			#A2 =  (1.0-math.exp((b-self.r)*self.T)*norm.cdf(d1))* (S_star/q2)
			#C=c+A2*math.pow((self.S/S_star),q2)

		return max(C,c) # know value will never be less than BS value

	def FiniteDifferenceImplicit(self, option):
		pass

	def FiniteDifferenceExplicit(self, option):
		pass

	def CrankNicolson(self, option):
		# Special Type of finite difference algorithm
		pass

	def ProjectedSOR(self, option):
		pass


	def __comb(self, n, steps, type):
		dt = round(self.T/steps, 5)
		u = math.exp(self.sigma * math.sqrt(dt))
		d = 1/u
		move = np.power(d, xrange(n, -1, -1)) * np.power(u, xrange(n + 1))
		if type.lower() == "call":
			val = self.S * move - self.X
		elif type.lower() == "put":
			val = self.X - self.S * move
		else:
			raise ValueError('Value Entered Is Incorrect')
		return np.maximum(val, 0)

	def w(self, n, steps, type):
		dt = round(float(self.T)/steps, 5)
		u = math.exp(self.sigma * math.sqrt(dt))
		d = 1/u
		discount = math.exp(-self.r * dt)
		p = (math.exp(self.r * dt) - d) / (u - d)

		leaf = self.__comb(n, steps, type)

		for _ in xrange(n, 0, -1):
			parent = leaf[:-1]/d
			leaf = discount * (p * leaf[1:] + (1 - p) * leaf[:-1])
			leaf = np.maximum(leaf, parent)

		return leaf[0]

	'''
	def other(self, option, n):
		self.T = float(self.T)
		dt = self.T/n
		up = np.exp(self.sigma * np.sqrt(dt))
		p0 = (up * np.exp(-self.d * dt) - np.exp(-self.r * dt)) / (up**2 - 1)
		p1 = np.exp(-self.r * dt) - p0
		p = np.zeros(n)

		'initial values at time T'
		for i in xrange(0, n):
			p[i] = (self.X - self.S) * up**(2*i - n)
			if p[i] < 0:
				p[i] = 0

		'move to earlier times'
		for j in range(n-1, -1, -1):
			for i in range(0, j):
				p[i] = p0 * p[i + 1] + p1 * p[i] # Binomial Value
				exercise = (self.X - self.S) * up**(2*i - j)
				if p[i] < exercise:
					p[i] = exercise

		return p[0]
	'''

if __name__ == "__main__":

	AM = American(S=100, X=90, r=0.05, sigma=0.36, T=2)

	import time
	for _ in xrange(10):
		start = time.time()
		#print "CRR American: " + str(AM.CRR("Call", 1000))
		#print "Quad Approx Put American: " + str(AM.QuadApproxPut())
		print "Fast Binomial:", str(AM.w(1000, 1000, "call"))
		end = time.time()
		print "Time: " + str(end - start)
