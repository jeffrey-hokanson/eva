#!/usr/bin/env python
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
import numpy as np
import matplotlib.pyplot as plt

def em_algorithm(x):
    pass

def kde(x):
    sigma = 1
    xx = np.linspace(-1, 15, 1000)
    yy = 0*xx
    for pt in x:
        yy += 1/(np.sqrt(2*np.pi*sigma))*np.exp(-(xx-pt)**2/(2*sigma**2))
    return xx, yy

# Build fake data
mu = [0, 10]
sigma = [1, 4]
n = [10000, 10000]

x = []
for mu_, sigma_, n_ in zip(mu, sigma, n):
    x.append(np.random.normal(mu_, sigma_, (n_,) ))

x = np.array(x).flatten()
print x
print x.shape
plt.plot(x, np.zeros(x.shape), '+')
xx, yy = kde(x)
plt.plot(xx, yy, 'r-')

# true pde
yy2 = 0*yy
for mu_, sigma_, n_ in zip(mu, sigma, n):
    yy2 += n_*1/(np.sqrt(2*np.pi*sigma_))*np.exp(-(xx-mu_)**2/(2*sigma_**2))

plt.plot(xx, yy2, 'b-')
plt.show()
