import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
%matplotlib inline


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def leak_relu(data, epsilon=0.1):
    return np.maximum(epsilon * data, data)


def relu(data):
    return np.maximum(0, data)


def Linear1(x, W=1.5):
    return W * x


x = np.arange(-2, 2, 0.01)
y_linear1 = Linear1(x)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_leak_relu = leak_relu(x)
y_relu = relu(x)


plt.plot(x, y_linear1, color='#1797ff', linewidth=3)
ax = plt.gca()
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.axhline(0, linestyle='--', color='gray', linewidth=1)
ax.axvline(0, linestyle='--', color='gray', linewidth=1)
plt.title("Linear")
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
