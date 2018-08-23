import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
import matplotlib


matplotlib.rcParams['figure.figsize'] = (10., 7.)


###############################################
# Get the orginal data form boston dataset
###############################################
lr = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target
predicted = cross_val_predict(lr, boston.data, y, cv=10)

plt.figure(1)
plt.subplot(221)
plt.scatter(y, predicted, s=10)
ax = plt.gca()


###############################################
# Combine the y and predicted data into one matrix be the orginal input data
###############################################
B = np.array([y]).T
C = np.array([predicted]).T
D = np.hstack((B, C))

plt.figure(1)
plt.subplot(222)
plt.scatter(D[:, 0], D[:, 1], s=10)
ax = plt.gca()
ax.axhline(0, linestyle='--', color='black', linewidth=1)
ax.axvline(0, linestyle='--', color='black', linewidth=1)
plt.title("Original Data")


###############################################
# Means data
###############################################
D -= np.mean(D, axis=0)
# Dmeans -= Dmeans

plt.figure(1)
plt.subplot(223)
ax = plt.gca()
ax.axhline(0, linestyle='--', color='black', linewidth=1)
ax.axvline(0, linestyle='--', color='black', linewidth=1)
ax.set_xlim([-40, 40])
ax.set_ylim([-25, 25])
plt.scatter(D[:, 0], D[:, 1], s=20)
plt.title("Zero Centered data")


###############################################
# normalized data
###############################################
D /= np.std(D, axis=0)
# Dstd -= Dstd

plt.figure(1)
plt.subplot(224)
ax = plt.gca()
ax.axhline(0, linestyle='--', color='black', linewidth=1)
ax.axvline(0, linestyle='--', color='black', linewidth=1)
ax.set_xlim([-40, 40])
ax.set_ylim([-25, 25])
plt.scatter(D[:, 0], D[:, 1], s=20)
plt.title("Normalized data")


plt.show()
