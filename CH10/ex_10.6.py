import numpy as np
from scipy.stats import norm


n = 1919
x0 = 922
x1 = n - x0
alpha = 0.05
theta0 = 0.5

z = norm.ppf(1 - alpha / 2)
print(z)

x_hat = x0
p_hat = x0 / n
x_exp = theta0 * n
se_hat = np.sqrt(n * (p_hat) * (1 - p_hat))

print((x_hat - x_exp) / se_hat)


# 11.2 exercise
import matplotlib.pyplot as plt
n = 1000
mu = 5
sigma = 1
x = np.random.normal(mu, sigma, size=n)

plt.hist(x)
plt.show()
