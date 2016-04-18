"""
==================================================================
Plot 1-D acquisition values as a function of number of iterations.
==================================================================
"""
print(__doc__)
import numpy as np
import matplotlib.pyplot as plt

from skopt import gp_minimize
from skopt.benchmarks import bench3
from skopt.gp_opt import acquisition

bounds = [[-2, 2]]
x = np.linspace(-2, 2, 200)
func_values = [bench3(xi) for xi in x]

vals = np.reshape(x, (-1, 1))

for n_iter in [10, 20]:
    res = gp_minimize(
        bench3, bounds, search='sampling', maxiter=n_iter,
        random_state=0, acq='LCB', n_start=5)
    gp_model = res.models[-1]
    best_x_l = res.x_iters.ravel()

    posterior_mean, posterior_std = gp_model.predict(vals, return_std=True)
    acquis_values = acquisition(vals, gp_model, method="LCB")
    posterior_mean = posterior_mean.ravel()
    posterior_std = posterior_std.ravel()
    upper_bound = posterior_mean + posterior_std
    lower_bound = posterior_mean - posterior_std

    plt.plot(x, posterior_mean, linestyle="--", label="Posterior mean", color='red')
    plt.plot(x, func_values, label="True values", color='green')
    plt.fill_between(
        x, lower_bound, upper_bound, alpha=0.3, label="Interval", color='blue')
    plt.plot(x, acquis_values, label="LCB values", color='black')

    sampled_y = [bench3(x) for x in best_x_l]
    plt.plot(best_x_l, sampled_y, 'ro')

    plt.legend(loc="best")
    plt.title("GP based minimization at n_iter = %d" % n_iter)
    plt.show()
