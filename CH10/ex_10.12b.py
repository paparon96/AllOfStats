import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt

def poisson_wald_test(n, lambda0, alpha):
    x = np.random.poisson(lambda0, size=n)
    x_bar = np.mean(x)
    se = np.sqrt(np.mean(x) / n)

    if abs((x_bar - 1) / se) > norm.ppf(1 - alpha / 2):
        return True
    else:
        return False

n = 20
lambda0 = 1
B = 1000
alpha = 0.05

z = norm.ppf(1 - alpha / 2)
print(z)
x = np.random.poisson(lambda0, size=n)
x_bar = np.mean(x)
print(x_bar)
print(np.var(x))

se = np.sqrt(np.mean(x) / n)
print(se)

# Run the simulations
test_results = np.zeros(B).astype(bool)
for b in range(B):
    test_results[b] = poisson_wald_test(n, lambda0, alpha)

print("######"*5)
print(f"Theoretical type 1 error: {alpha}")
print(f"Estimated rejection ratio: {sum(test_results) / len(test_results)}")
