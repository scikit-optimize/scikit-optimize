import time
import matplotlib.pyplot as plt
from ProcessOptimizer.benchmarks import branin as branin
from ProcessOptimizer.benchmarks import hart6 as hart6_
from ProcessOptimizer.plots import plot_objective
from ProcessOptimizer import gp_minimize, forest_minimize, dummy_minimize
from ProcessOptimizer import plots
from ProcessOptimizer import bokeh_plot
# For reproducibility
import numpy as np
np.random.seed(123)
plt.set_cmap("viridis")
# Here we define a function that we evaluate.


def funny_func(x):
    s = 0
    for i in range(len(x)):
        s += (x[i])**2
    return s


# We run forest_minimize on the function
bounds = [(-1, 1.), ] * 7
n_calls = 30

result = gp_minimize(funny_func, bounds, n_calls=n_calls, n_random_starts=20,
                     acq_optimizer="auto", acq_func="gp_hedge", random_state=4)

bokeh_plot.start(result)
