import numpy as np
import matplotlib.pyplot as plt


###############################################

###############################################
def neurous(sqrt=False):
    num = 1000
    x = np.ones(num)
    if not sqrt:
        w = np.random.randn(num)
    else:
        w = np.random.randn(num) / np.sqrt(num)
    b = 0
    z = np.dot(w, x) + b
    return z


zl = []
for i in range(200000):
    zl.append(neurous())

print(np.mean(zl))
print(np.var(zl))


###############################################

###############################################
n, bins, patches = plt.hist(zl, 100, alpha=0.8)
plt.xlabel('z')
plt.ylabel('Probability')
plt.title('Histogram of $z=wx+b$')
plt.show()


###############################################

###############################################
def sigmod(x): return 1. / 1. + np.exp(-x)


al = sigmod(np.array(zl))

n, bins, patches = plt.hist(al, 100, normed=1, alpha=0.8)
plt.xlabel('a')
plt.ylabel('Probability')
plt.title('Histogram of $a=sigmod(z)$')
plt.show()


###############################################

###############################################
zl = []
for i in range(200000):
    zl.append(neurous(sqrt=True))

n, bins, patches = plt.hist(zl, 100, alpha=0.8)
plt.xlabel('z')
plt.ylabel('Probability')
plt.title('Histogram of $z=wx+b$')
plt.show()


###############################################

###############################################
al = sigmod(np.array(zl))

n, bins, patches = plt.hist(al, 100, normed=1, alpha=0.8)
plt.xlabel('a')
plt.ylabel('Probability')
plt.title('Histogram of $a=sigmod(z)$')
plt.show()
