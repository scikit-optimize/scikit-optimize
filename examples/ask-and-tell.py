"""
=======================
Async optimization Loop
=======================
Bayesian optimization is used to tune parameters for walking robots or other
experiments that are not a simple (expensive) function call.

Tim Head, February 2017.
Reformatted by Holger Nahrstaedt 2020

.. currentmodule:: skopt

They often follow a pattern a bit like this:

1. ask for a new set of parameters
2. walk to the experiment and program in the new parameters
3. observe the outcome of running the experiment
4. walk back to your laptop and tell the optimizer about the outcome
5. go to step 1

A setup like this is difficult to implement with the ***_minimize()** function
interface. This is why **scikit-optimize** has a ask-and-tell interface that
you can use when you want to control the execution of the optimization loop.

This notebook demonstrates how to use the ask and tell interface.
"""
print(__doc__)

import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process

#############################################################################
# The Setup
# ---------
# We will use a simple 1D problem to illustrate the API. This is a little bit
# artificial as you normally would not use the ask-and-tell interface if you
# had a function you can call to evaluate the objective.

from skopt.learning import ExtraTreesRegressor
from skopt import Optimizer

noise_level = 0.1

#########################################################################
# Our 1D toy problem, this is the function we are trying to
# minimize

def objective(x, noise_level=noise_level):
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2))\
           + np.random.randn() * noise_level

def objective_wo_noise(x, noise_level=0):
    return objective(x, noise_level=0)

#########################################################################
# Here a quick plot to visualize what the function looks like:

# Plot f(x) + contours
plt.set_cmap("viridis")
x = np.linspace(-2, 2, 400).reshape(-1, 1)
fx = np.array([objective(x_i, noise_level=0.0) for x_i in x])
plt.plot(x, fx, "r--", label="True (unknown)")
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate(([fx_i - 1.9600 * noise_level for fx_i in fx],
                         [fx_i + 1.9600 * noise_level for fx_i in fx[::-1]])),
         alpha=.2, fc="r", ec="None")
plt.legend()
plt.grid()
plt.show()

#########################################################################
# Now we setup the :class:`Optimizer` class. The arguments follow the meaning and
# naming of the ***_minimize()** functions. An important difference is that
# you do not pass the objective function to the optimizer.

opt = Optimizer([(-2.0, 2.0)], "GP", acq_func="EI",
                acq_optimizer="sampling",
                initial_point_generator="lhs")

# To obtain a suggestion for the point at which to evaluate the objective
# you call the ask() method of opt:

next_x = opt.ask()
print(next_x)

#########################################################################
# In a real world use case you would probably go away and use this
# parameter in your experiment and come back a while later with the
# result. In this example we can simply evaluate the objective function
# and report the value back to the optimizer:

f_val = objective(next_x)
opt.tell(next_x, f_val)

#########################################################################
# Like ***_minimize()** the first few points are suggestions from
# the initial point generator as there
# is no data yet with which to fit a surrogate model.


for i in range(9):
    next_x = opt.ask()
    f_val = objective(next_x)
    res = opt.tell(next_x, f_val)

#########################################################################
# We can now plot the random suggestions and the first model that has been
# fit:
_ = plot_gaussian_process(res, objective=objective_wo_noise,
                          noise_level=noise_level,
                          show_next_point=False,
                          show_acq_func=True)
plt.show()
#########################################################################
# Let us sample a few more points and plot the optimizer again:


for i in range(10):
    next_x = opt.ask()
    f_val = objective(next_x)
    res = opt.tell(next_x, f_val)

_ = plot_gaussian_process(res, objective=objective_wo_noise,
                          noise_level=noise_level,
                          show_next_point=True,
                          show_acq_func=True)
plt.show()
#########################################################################
# By using the :class:`Optimizer` class directly you get control over the
# optimization loop.
#
# You can also pickle your :class:`Optimizer` instance if you want to end the
# process running it and resume it later. This is handy if your experiment
# takes a very long time and you want to shutdown your computer in the
# meantime:

import pickle

with open('my-optimizer.pkl', 'wb') as f:
    pickle.dump(opt, f)

with open('my-optimizer.pkl', 'rb') as f:
    opt_restored = pickle.load(f)
