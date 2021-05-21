import numpy as np
import matplotlib.pyplot as plt

from main import fit
from lmfit import Parameters


def func(pars, x, data):
    a, b, c = pars['a'], pars['b'], pars['c']
    model = a * np.exp(-b * x) + c
    if data is None:
        return model
    return model - data


def f(var, x):
    return var[0] * np.exp(-var[1] * x) + var[2]


params = Parameters()
params.add('a', value=10)
params.add('b', value=10)
params.add('c', value=10)

a, b, c = 2.5, 1.3, 0.8
x = np.linspace(0, 4, 50)
y = f([a, b, c], x)
data = y + 0.15 * np.random.normal(size=x.size)

# fit without analytic derivative
result = fit(func, params, args=(x, data), algo_choice="levmar")

plt.plot(x, data, 'b')
plt.plot(x, data + func(params, x, data), 'y')
plt.plot(x, data + func(result, x, data), 'r')
plt.show()
