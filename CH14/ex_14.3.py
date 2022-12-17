import numpy as np

import matplotlib.pyplot as plt

def multinom(n, p, nsim):

    return np.random.multinomial(n, p, size=nsim)

n = 100
k = 3
p = [1/k]*k
nsim = 500
x = multinom(n, p, nsim)
print(x.shape)

# plt.hist(x[:, 0])
# plt.show()



x_bar = np.mean(x, axis=0)
print(x_bar)

x_se = np.std(x, axis=0)
print(x_se)



# multivariate multivariate_normal
mu = np.random.uniform(size=k)
sigma = np.eye(k)

x = np.random.multivariate_normal(mu, sigma, size=nsim)
print(x.shape)

print(mu)

x_bar = np.mean(x, axis=0)
print(x_bar)
