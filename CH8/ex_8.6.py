import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt

n = 100
mu = 5
sigma = 1
B = 200
alpha = 0.05

z = norm.ppf(1 - alpha / 2)
print(z)
x = np.random.normal(mu, sigma, size=n)
x_bar = np.mean(x)
print(x_bar)

theta_hat = np.exp(x_bar)
print(theta_hat)

theta_hats = np.zeros(B)
for b in range(B):
    boot_sample = np.random.choice(x, size=n)
    boot_theta_hat = np.exp(np.mean(boot_sample))
    theta_hats[b] = boot_theta_hat

se = np.std(theta_hats)
print(se)

ci = (theta_hat-z*se, theta_hat+z*se)
print(ci)

plt.hist(theta_hats)
plt.show()
