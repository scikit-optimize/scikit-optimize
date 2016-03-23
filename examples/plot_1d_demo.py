import numpy as np
import math

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from skopt.gp_opt import gp_minimize

def parabola(x):
    return x[0]**2

def sin(x):
    return math.sin(x[0])

def scale_to_uniform(x, lower_bounds, upper_bounds):
    return x - lower_bounds / (upper_bounds - lower_bounds)

def plot_interactive_gp(func, bounds, random_state, max_iter=1000):
    rng = np.random.RandomState(0)
    x, func_val, d = gp_minimize(
        func, (bounds,), search='lbfgs', maxiter=max_iter, random_state=0,
        acq='UCB')
    gp_models = d["models"]
    best_x_l = d["x_iters"].ravel()

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.title("Gaussian Process Approximation")
    t = np.linspace(bounds[0], bounds[1], 10000)
    t_gp = scale_to_uniform(t, bounds[0], bounds[1])
    t_gp = t_gp.reshape(-1, 1)

    y = [func([ele]) for ele in t]
    l, = plt.plot(t, y, lw=2, color='green')
    l1, = plt.plot(t, y, 'r--', lw=2)
    point = plt.plot([0], [0], 'ro')

    plt.axis([bounds[0], bounds[1], np.min(y), np.max(y)])

    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)

    gp_iter = Slider(axfreq, 'Iterations', 0, max_iter, valinit=1, valfmt="%d")

    def update(val):
        i = int(gp_iter.val)
        l1.set_ydata(gp_models[i - 1].predict(t_gp))
        point[-1].set_xdata(best_x_l[i - 1])
        point[0].set_ydata(func([best_x_l[i - 1]]))
        fig.canvas.draw_idle()

    gp_iter.on_changed(update)

    plt.show()
    return x, func_val, d

plot_interactive_gp(parabola, (-1, 1), 0, 100)
# x, f, d = plot_interactive_gp(sin, (-2, 2), 0, 100)
