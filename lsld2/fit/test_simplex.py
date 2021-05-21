import matplotlib.pyplot as plt
import numpy as np

from main import fit
from lmfit import Parameters

x = np.linspace(1, 10, 250)
np.random.seed(0)
y = 3.0 * np.exp(-x / 2) - 5.0 * np.exp(-(x - 0.1) / 10.) + 0.1 \
    * np.random.randn(x.size)

p = Parameters()
p.add_many(('a1', 4.), ('a2', 4.), ('t1', 3.), ('t2', 3., True))


def residual(p):
    v = p
    return v['a1'] * np.exp(-x / v['t1']) + v['a2'] * np.exp(-(x - 0.1)
                                                             / v['t2']) - y


result = fit(residual, p, algo_choice="simplex")

plt.plot(x, y, 'b')
plt.plot(x, y + residual(p), 'y')
plt.plot(x, y + residual(result), 'r')
plt.show()
