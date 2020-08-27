import numpy as np
import math
import time

class OptionPrice:
    def __init__(self, param):
        self.pc = param.pc
        self.S = param.S
        self.K = param.K
        self.T = param.T
        self.r = param.r
        self.sigma = param.sigma
        self.iterations = param.iterations

    def getMCPrice(self):
        'Standard Monte-Carlo Approach'
        # create placeholder array or size n
        hold = np.zeros([self.iterations, 2])
        # generate n numbers from a standard normal dist
        rand = np.random.normal(0, 1, [1, self.iterations])

        # Monte Carlo equation
        mult = self.S * np.exp(self.T * (self.r - 0.5 * self.sigma ** 2))
        if self.pc == 'call': # use call equation
            hold[:, 1] = mult * np.exp(np.sqrt((self.sigma ** 2) * self.T) * rand) - self.K
        elif self.pc == 'put': # use put equation
            hold[:, 1] = self.K - mult * np.exp(np.sqrt((self.sigma ** 2) * self.T) * rand)

        # calculate average
        avg_po = np.sum(np.amax(hold, axis=1)) / float(self.iterations)

        # lastly, multiply by e^(-rT) as specified in Monte Carlo equation
        return np.exp(-1.0 * self.r * self.T) * avg_po

    def getBlackScholesPrice(self):
        'Exact Option Price using Black-Scholes equation.'
        # calculate d1 and d2
        d1 = (np.log(self.S / self.K) + (self.r + self.sigma ** 2 / 2) * self.T)
        d1 /= self.sigma * np.sqrt(self.T)
        d2 = d1 - self.sigma * np.sqrt(self.T)

        # find the cdf probability of d1 and d2
        ncdf_d1 = (1.0 + math.erf(d1 / math.sqrt(2.0))) / 2.0
        ncdf_d2 = (1.0 + math.erf(d2 / math.sqrt(2.0))) / 2.0

        # plug back to Black-Scholes Equation
        call = (self.S * ncdf_d1) - (self.K * np.exp(-1.0 * self.r * self.T) * ncdf_d2)

        if self.pc == 'call': # return call as usual if call option
            return call
        elif self.pc == 'put': # return the value from PCParity function
            return self.PCParity(call)

    def PCParity(self, call):
        'Use Put-Parity Relation to determine put price.'
        return call - self.S + self.K * np.exp(-1.0 * self.r * self.T)

    def getMCPrice_Antithetic(self):
        'Monte-Carlo Approach with Antithetic Variables'

        # create placeholder array or size n
        hold = np.zeros([self.iterations, 2])
        # generate n/2 numbers from a standard normal dist
        rand = np.random.normal(0, 1, [1, self.iterations/2])

        # Monte Carlo equation
        mult = self.S * np.exp(self.T * (self.r - 0.5 * self.sigma ** 2))
        temp = mult * np.exp(np.sqrt((self.sigma ** 2) * self.T) * rand) - self.K
        # Monte Carlo equation with negative random variable to satisfy Antithetic method
        temp_inv = mult * np.exp(np.sqrt((self.sigma ** 2) * self.T) * -rand) - self.K

        # append to a single array, and replace the placeholder array
        hold[:, 1] = np.append(temp, temp_inv, axis=1)
        # calculate average
        avg_po = np.sum(np.amax(hold, axis=1)) / float(self.iterations)

        # lastly, multiply by e^(-rT) as specified in Monte Carlo equation
        return np.exp(-1.0 * self.r * self.T) * avg_po

    def getMCPrice_CV(self):
        'Monte-Carlo Approach with Control Variates'

        # Vanilla Monte Carlo for call option
        c = self.getMCPrice()

        # Exact put option price using Black-Scholes
        self.pc = 'put'
        p_bs = self.getBlackScholesPrice()
        # Vanilla Monte Carlo for put option
        p = self.getMCPrice()
        self.pc = 'call'

        return c + p_bs - p


