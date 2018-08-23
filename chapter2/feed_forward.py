import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
%matplotlib inline


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def feedForward(X, W, b, hidden=2):
    ah = [1.0] * hidden

    for j in range(hidden):
        sum = 0.0
        for i in range(input):
            sum += X[i] * w1[i][j]

        ah[j] = sigmoid(sum + b[j])
    return ah


X = np.array([1, 2], dtype=float)
W = np.array([[1, -1], [-2, 1]], dtype=float)
b = np.array([1, 0], dtype=float)
ah = feedForward(X, W, b)
print(ah)


plt.plot(x, y_linear1, color='#1797ff', linewidth=3)
ax = plt.gca()
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.axhline(0, linestyle='--', color='gray', linewidth=1)
ax.axvline(0, linestyle='--', color='gray', linewidth=1)
plt.title("Linear")
plt.show()


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


y_softmax = softmax(x)

plt.plot(x, y_softmax, color='#1797ff', linewidth=3)
ax = plt.gca()
ax.set_xlim([-2, 2])
ax.set_ylim([-0.001, 0.01])
ax.axhline(0, linestyle='--', color='gray', linewidth=1)
ax.axvline(0, linestyle='--', color='gray', linewidth=1)
plt.title("Softmax")
plt.show()

plt.plot(x, y_sigmoid, color='#1797ff', linewidth=3)
ax = plt.gca()
ax.set_xlim([-2, 2])
ax.set_ylim([-1, 1])
ax.axhline(0, linestyle='--', color='gray', linewidth=1)
ax.axvline(0, linestyle='--', color='gray', linewidth=1)
plt.title("sigmoid")
plt.show()


plt.plot(x, y_tanh, color='#1797ff', linewidth=3)
ax = plt.gca()
ax.set_xlim([-2, 2])
ax.set_ylim([-1.1, 1.1])
ax.axhline(0, linestyle='--', color='gray', linewidth=1)
ax.axvline(0, linestyle='--', color='gray', linewidth=1)
plt.title("tanh")
plt.show()


plt.plot(x, y_leak_relu, color='#1797ff', linewidth=3)
ax = plt.gca()
ax.set_xlim([-2, 2])
ax.set_ylim([-0.5, 2])
ax.axhline(0, linestyle='--', color='gray', linewidth=1)
ax.axvline(0, linestyle='--', color='gray', linewidth=1)
plt.title("leak ReLU")
plt.show()


plt.plot(x, y_relu, color='#1797ff', linewidth=3)
ax = plt.gca()
ax.set_xlim([-2, 2])
ax.set_ylim([-0.5, 2])
ax.axhline(0, linestyle='--', color='gray', linewidth=1)
ax.axvline(0, linestyle='--', color='gray', linewidth=1)
plt.title("ReLU")
plt.show()
