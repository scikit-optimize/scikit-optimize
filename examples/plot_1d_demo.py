import numpy as np
import math

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from skopt import acquisition
from skopt import gp_minimize


def parabola(x):
    return x[0] ** 2


def sin(x):
    return math.sin(x[0])


def plot_interactive_gp(func, bounds, random_state, max_iter=1000):
    rng = np.random.RandomState(0)
    res = gp_minimize(
        func, (bounds,), search='lbfgs', maxiter=max_iter, random_state=0,
        acq='UCB')
    gp_models = res.models
    best_x_l = res.x_iters.ravel()

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.title("Gaussian Process Approximation")
    t = np.linspace(bounds[0], bounds[1], 10000)
    t = np.reshape(t, (10000, -1))

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
        y_data = acquisition(t, gp_models[i - 1], bounds=[bounds])
        l1.set_ydata(y_data)
        point[-1].set_xdata(best_x_l[i - 1])
        point[0].set_ydata(func([best_x_l[i - 1]]))
        fig.canvas.draw_idle()

    gp_iter.on_changed(update)

    plt.show()

plot_interactive_gp(parabola, (-3, 3), 0, 100)
