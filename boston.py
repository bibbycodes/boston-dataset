import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

from matplotlib import rcParams
from sklearn.datasets import load_boston
boston = load_boston()


bos = pd.DataFrame(boston.data) # load sklearn dataset into pandas DF object
bos.columns = boston.feature_names # convert features to pandas
# print(boston.target.shape)
# print(bos.head())

# Create axes

# X = bos.drop('MEDV', axis = 1)
print(bos.keys)
# Y = bos['MEDV']

#  separate into
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
print(bos.describe())