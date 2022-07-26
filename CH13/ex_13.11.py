import pandas as pd
import numpy as np

import statsmodels.api as sm

import matplotlib.pyplot as plt

# url = "https://www.stat.cmu.edu/~larry/all-of-statistics/=data/coris.dat"
# res = requests.get(url)
# # pd.read_csv(io.BytesIO(res.content), sep=';')
#
# # df = pd.read_fwf(io.BytesIO(res.content))
# df = pd.read_fwf(url)
#
# print(df.head())


df = pd.read_fwf("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
'acceleration', 'model_year', 'origin', 'name']

df['bin_target'] = (df['mpg'] > np.mean(df['mpg'])).astype(int)

X = sm.add_constant(df[['displacement', 'weight']].astype(float).copy())
Y = df['bin_target'].copy()


# building the model and fitting the data
log_reg = sm.Logit(Y, X).fit()

print(log_reg.summary())
