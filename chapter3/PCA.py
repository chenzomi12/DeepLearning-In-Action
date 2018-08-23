import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
import matplotlib

matplotlib.rcParams['figure.figsize'] = (10., 7.)


# Get the orginal data form boston dataset
lr = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target
predicted = cross_val_predict(lr, boston.data, y, cv=10)

plt.figure(1)
plt.subplot(221)
plt.scatter(y, predicted, s=10)
ax = plt.gca()


# Combine the y and predicted data into one matrix be the orginal input data
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


# Means data
D -= np.mean(D, axis=0)
cov = np.dot(D.T, D) / D.shape[0]
U, S, V = np.linalg.svd(cov)
Xrot = np.dot(D, U)
# Dmeans -= Dmeans

plt.figure(1)
plt.subplot(223)
ax = plt.gca()
ax.axhline(0, linestyle='--', color='black', linewidth=1)
ax.axvline(0, linestyle='--', color='black', linewidth=1)
ax.set_xlim([-40, 40])
ax.set_ylim([-25, 25])
plt.scatter(Xrot[:, 0], Xrot[:, 1], s=20)
plt.title("PCA data")


# normalized data
Xwhite = Xrot / np.sqrt(S + 1e-5)  # Dstd -= Dstd

plt.figure(1)
plt.subplot(224)
ax = plt.gca()
ax.axhline(0, linestyle='--', color='black', linewidth=1)
ax.axvline(0, linestyle='--', color='black', linewidth=1)
ax.set_xlim([-40, 40])
ax.set_ylim([-25, 25])
plt.scatter(Xwhite[:, 0], Xwhite[:, 1], s=20)
plt.title("Whitened data")
plt.show()
