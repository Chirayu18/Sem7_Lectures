import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import scipy.optimize as fit
from numpy.random import default_rng

npoints = 20
params = [1, -121, 3688, -6900]
# params = [1, -2, 1]

rng = default_rng(4492)
errors = rng.standard_normal(npoints)

x = np.linspace(0, 100, npoints)
Fgen = []
for i in x:
    Fgen.append([i**k for k in range(len(params) - 1, -1, -1)])

F = np.array(Fgen)
# I
# Want
# To
# Make the number of lines 69


def f(m):
    return np.dot(F, m)


y = np.array([])
y = np.append(y, f(params))
data = y + errors * 0.3 * (y)

R = np.diag([abs(i) for i in data - np.dot(F, params)])
Ft = np.linalg.multi_dot(
    [np.linalg.inv(np.linalg.multi_dot([np.transpose(F), R, F])), np.transpose(F), R]
)


def fit(data):
    return np.dot(Ft, data)


fits = fit(data)
print("Fitted Parameters:")
print(fits)
print("Model Resolution matrix")
print(np.dot(Ft, F))
print("Data Resolution matrix")
print(np.dot(F, Ft))
ax = sns.heatmap(np.dot(F, Ft))
plt.savefig("drm.png")
plt.clf()
ax = sns.heatmap(np.dot(Ft, F))
plt.savefig("mrm.png")
plt.clf()
plt.plot(x, y, "-o", label="original")
plt.plot(x, data, "-o", label="original + errors")
plt.plot(x, f(fits), "-o", label="fiited L1")
R = np.identity(len(y))
Ft = np.linalg.multi_dot(
    [np.linalg.inv(np.linalg.multi_dot([np.transpose(F), R, F])), np.transpose(F), R]
)
fits = fit(data)
plt.plot(x, f(fits), "-o", label="fiited")
plt.legend()
plt.savefig("plot.png")
plt.show()
