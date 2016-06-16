"""
=============================================================
Lower confidence bound as a function of number of iterations.
=============================================================

``gp_minimize`` approximates a costly function with a GaussianProcess prior
and instead of minimizing this costly function, it minimizes an acquisition
function which is relatively cheaper to minimize. Before every iteration of
``gp_minimize``, a fixed number of points are used to approximate the
GaussianProcess prior. The acquisition function combines information
about

1. New points that are close to the previous points used to approximate the GP
prior and have a very low expected value. (Posterior mean)
2. New points that are far away from these previous points and hence have a
very high uncertainty. (Posterior std)

The plot shows the "Lower Confidence Bound" after 2, 5 and 10 iterations of
``gp_minimize`` with the branin function. It may be worth noting that the areas
around the points previously sampled have very low std values.
"""
print(__doc__)
import numpy as np
import matplotlib.pyplot as plt

from skopt import gp_minimize
from skopt.benchmarks import branin
from skopt.acquisition import gaussian_lcb

fig = plt.gcf()
plt.set_cmap("viridis")
dimensions = [(-5.0, 10.0), (0.0, 15.0)]

# True minima of the branin function
min_x = [-np.pi, np.pi, 9.42478]
min_y = [12.275, 2.275, 2.475]

x1_values = np.linspace(-5, 10, 100)
x2_values = np.linspace(0, 15, 100)
x_ax, y_ax = np.meshgrid(x1_values, x2_values)
vals = np.c_[x_ax.ravel(), y_ax.ravel()]
subplot_no = 221

res = gp_minimize(
    branin, dimensions, search='lbfgs', maxiter=10, random_state=0,
    acq="LCB", n_start=1, n_restarts_optimizer=2)
gp_model = res.models[-1]
opt_points = res['x_iters']

posterior_mean, posterior_std = gp_model.predict(vals, return_std=True)
acquis_values = gaussian_lcb(vals, gp_model)
acquis_values = acquis_values.reshape(100, 100)
posterior_mean = posterior_mean.reshape(100, 100)
posterior_std = posterior_std.reshape(100, 100)
best_min = vals[np.argmin(acquis_values)]

plt.subplot(subplot_no)
plt.pcolormesh(x_ax, y_ax, posterior_mean)
plt.plot(opt_points[:, 0], opt_points[:, 1], 'wo', markersize=5, label="sampled points")
plt.plot(best_min[0], best_min[1], 'ro', markersize=5, label="GP min")
plt.plot(min_x, min_y, 'go', markersize=5, label="true minima")
plt.colorbar()
plt.xlabel('X1')
plt.xlim([-5, 10])
plt.ylabel('X2')
plt.ylim([0, 15])
plt.title("Posterior mean for acq=LCB")
plt.legend(loc="best", prop={'size': 8}, numpoints=1)
subplot_no += 1

plt.subplot(subplot_no)
plt.pcolormesh(x_ax, y_ax, posterior_std)
plt.plot(opt_points[:, 0], opt_points[:, 1], 'wo', markersize=5)
plt.plot(best_min[0], best_min[1], 'ro', markersize=5)
plt.plot(min_x, min_y, 'go', markersize=5)
plt.colorbar()
plt.xlabel('X1')
plt.xlim([-5, 10])
plt.ylabel('X2')
plt.ylim([0, 15])
plt.title("Posterior std for acq=LCB")
subplot_no += 1

plt.subplot(subplot_no)
plt.pcolormesh(x_ax, y_ax, acquis_values)
plt.plot(opt_points[:, 0], opt_points[:, 1], 'wo', markersize=5)
plt.plot(best_min[0], best_min[1], 'ro', markersize=5)
plt.plot(min_x, min_y, 'go', markersize=5)
plt.colorbar()
plt.xlabel('X1')
plt.xlim([-5, 10])
plt.ylabel('X2')
plt.ylim([0, 15])
plt.title("LCB after 20 iterations.")
subplot_no += 1

plt.subplot(subplot_no)
func_values = np.reshape([branin(val) for val in vals], (100, 100))
plt.plot(opt_points[:, 0], opt_points[:, 1], 'wo', markersize=5)
plt.plot(best_min[0], best_min[1], 'ro', markersize=5)
plt.plot(min_x, min_y, 'go', markersize=5)
plt.pcolormesh(x_ax, y_ax, func_values)
plt.colorbar()
plt.xlabel('X1')
plt.xlim([-5, 10])
plt.ylabel('X2')
plt.ylim([0, 15])
plt.title("Function values")

plt.suptitle("2-D acquisition values on the branin function")
plt.show()
