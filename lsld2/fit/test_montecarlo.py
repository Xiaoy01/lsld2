from main import fit

import matplotlib.pyplot as plt
import numpy as np

from lmfit import Parameters


x = np.linspace(0, 15, 301)
np.random.seed(7)
noise = np.random.normal(size=x.size, scale=0.2)
data = (5. * np.sin(2 * x - 0.1) * np.exp(-x * x * 0.025) + noise)
plt.plot(x, data, 'b')


def fcn2min(params, x, data):
    """Model decaying sine wave, subtract data."""
    amp = params['amp']
    shift = params['shift']
    omega = params['omega']
    decay = params['decay']
    model = amp * np.sin(x * omega + shift) * np.exp(-x * x * decay)
    return model - data


# create a set of Parameters
params = Parameters()
# params.add('amp', value=7, min=2.5)
params.add('amp', value=7, min=0, max=14)
params.add('decay', value=0.05, min=0, max=0.1)
params.add('shift', value=0.0, min=-np.pi / 2., max=np.pi / 2)
params.add('omega', value=3, min=0, max=5)

params['amp'].set(brute_step=0.25)
params['decay'].set(brute_step=0.005)
params['omega'].set(brute_step=0.25)

params.pretty_print()

result = fit(fcn2min, params, args=(x, data), algo_choice='montecarlo')

plt.plot(x, data, 'b')
plt.plot(x, data + fcn2min(params, x, data), 'y')
plt.plot(x, data + fcn2min(result, x, data), 'r')
plt.show()
