import pandas as pd
import numpy as np

import statsmodels.api as sm

import matplotlib.pyplot as plt

df = pd.read_fwf("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
'acceleration', 'model_year', 'origin', 'name']

df['log_mpg'] = np.log(df['mpg'])

print(df.head())

# plt.hist(df['mpg'])
# plt.show()
# plt.hist(df['log_mpg'])
# plt.show()


######## MPG
# Data preprocessing
X = sm.add_constant(df[['displacement', 'weight']].astype(float).copy())
Y = df['mpg'].copy()

# OLS model
model = sm.OLS(Y,X)
result = model.fit()
print(result.summary())

Y_hat = result.predict(X)
plt.scatter(Y_hat, Y)
plt.xlabel("Prediction for MPG")
plt.ylabel("MPG")
plt.show()

plt.hist(Y_hat - Y)
plt.title("Residual error distribution")
plt.show()

######## LOG MPG
# Data preprocessing
X = sm.add_constant(df[['displacement', 'weight']].astype(float).copy())
Y = df['log_mpg'].copy()

# OLS model
model = sm.OLS(Y,X)
result = model.fit()
print(result.summary())


Y_hat = result.predict(X)
plt.scatter(Y_hat, Y)
plt.xlabel("Prediction for log MPG")
plt.ylabel("Log MPG")
plt.show()

plt.hist(Y_hat - Y)
plt.title("Residual error distribution")
plt.show()
