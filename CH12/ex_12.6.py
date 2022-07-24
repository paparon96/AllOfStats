import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt

def mle(x):
    return np.mean(x, axis=0)

def jse(x):
    return np.median(x, axis=0)

def risk(theta_hat, theta):
    return np.sum((theta_hat-theta)**2)

n = 100
k = 5
theta = np.random.uniform(size=k)
sigma = np.eye(k)
B = 200

# Single test
x = np.random.multivariate_normal(theta, sigma, size=n)
print(x.shape)
x_bar = np.mean(x, axis=0)
print(x_bar)

print(risk(x_bar, theta))

# experiment
mle_risks = np.zeros(B)
jse_risks = np.zeros(B)
for b in range(B):
    x = np.random.multivariate_normal(theta, sigma, size=n)

    mle_risks[b] = risk(mle(x), theta)
    jse_risks[b] = risk(jse(x), theta)

plt.hist(mle_risks, label="MLE", histtype='step')
plt.hist(jse_risks, label="JSE", histtype='step')
plt.legend()
plt.show()
