import numpy as np
import matplotlib.pyplot as plt

n = 50
theta = 1
B = 2000

x = np.random.uniform(0, theta, size=n)
theta_hat = np.max(x)
print(theta_hat)

theta_hats = np.zeros(B)
for b in range(B):
    boot_sample = np.random.choice(x, size=n)
    boot_theta_hat = np.max(boot_sample)
    theta_hats[b] = boot_theta_hat

se = np.std(theta_hats)
print(se)

# ci = (theta_hat-z*se, theta_hat+z*se)
# print(ci)

plt.hist(theta_hats, density=True)
plt.show()
