"""
==========================
Use unique base estimators
==========================

Sigurd Carlen, September 2019.
Reformatted by Holger Nahrstaedt 2020

.. currentmodule:: skopt


To use different base_estimator or create a regressor with different parameters,
we can create a regressor object and set it as kernel.

"""
print(__doc__)

import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as plt


#############################################################################
# Toy example
# -----------
#
# Let assume the following noisy function :math:`f`:

noise_level = 0.2

# Our 1D toy problem, this is the function we are trying to
# minimize
def objective(X, noise_level=noise_level):
    return -np.sin(3*X[0]) - X[0]**2 + 0.7*X[0] + noise_level * np.random.randn()

#############################################################################

from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
# Gaussian process with Matérn kernel as surrogate model
m52 = ConstantKernel(1.0) * Matern(length_scale=1.0,
                                   nu=2.5)
gpr = GaussianProcessRegressor(kernel=m52, alpha=noise_level**2,
                               normalize_y=True, noise="gaussian",
                               n_restarts_optimizer=2
                               )
#############################################################################

from skopt import Optimizer
opt = Optimizer([(-1.0, 2.0)], base_estimator=gpr, n_initial_points=5,
                acq_optimizer="sampling")
#############################################################################

x = np.linspace(-1, 2, 400).reshape(-1, 1)
fx = np.array([objective(x_i, noise_level=0.0) for x_i in x])

#############################################################################

from skopt.acquisition import gaussian_ei


def plot_optimizer(opt, x, fx):
    model = opt.models[-1]
    x_model = opt.space.transform(x.tolist())
    # Plot true function.
    plt.plot(x, fx, "r--", label="True (unknown)")
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([fx - 1.9600 * noise_level,
                             fx[::-1] + 1.9600 * noise_level]),
             alpha=.2, fc="r", ec="None")

    # Plot Model(x) + contours
    y_pred, sigma = model.predict(x_model, return_std=True)
    plt.plot(x, y_pred, "g--", label=r"$\mu(x)$")
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                             (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.2, fc="g", ec="None")

    # Plot sampled points
    plt.plot(opt.Xi, opt.yi,
             "r.", markersize=8, label="Observations")

    acq = gaussian_ei(x_model, model, y_opt=np.min(opt.yi))
    # shift down to make a better plot
    acq = 4 * acq - 2
    plt.plot(x, acq, "b", label="EI(x)")
    plt.fill_between(x.ravel(), -2.0, acq.ravel(), alpha=0.3, color='blue')

    # Adjust plot layout
    plt.grid()
    plt.legend(loc='best')

#############################################################################

for i in range(10):
    next_x = opt.ask()
    print("%.2f next x" % next_x[0])
    f_val = objective(next_x)
    r = opt.tell(next_x, f_val)
    if i >= 5:
        # plt.subplot(5, 1, i-4)
        plt.figure()
        plt.title("%d" % i)
        plot_optimizer(opt, x, fx)


#############################################################################

def plot_convergence(X_sample, Y_sample, n_init=5):
    plt.figure(figsize=(12, 3))

    x = X_sample[n_init:].ravel()
    y = Y_sample[n_init:].ravel()
    r = range(1, len(x) + 1)

    x_neighbor_dist = [np.abs(a - b) for a, b in zip(x, x[1:])]
    y_max_watermark = np.maximum.accumulate(y)

    plt.subplot(1, 2, 1)
    plt.plot(r[1:], x_neighbor_dist, 'bo-')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Distance between consecutive x\'s')

    plt.subplot(1, 2, 2)
    plt.plot(r, y_max_watermark, 'ro-')
    plt.xlabel('Iteration')
    plt.ylabel('Best Y')
    plt.title('Value of best selected sample')


plot_convergence(np.array(r.x_iters), -r.func_vals)

