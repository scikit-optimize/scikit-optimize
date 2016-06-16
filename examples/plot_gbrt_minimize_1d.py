"""
================================================================
Plot expected improvement as a function of number of iterations.
================================================================
"""
print(__doc__)
import numpy as np
import matplotlib.pyplot as plt

from skopt import gbrt_minimize
from skopt.benchmarks import bench3
from skopt.acquisition import gaussian_ei


dimensions = [(-1.0, 1.0)]
x = np.linspace(-1, 1, 200)
func_values = [bench3(xi) for xi in x]
plt.figure(figsize=(10, 5))
vals = np.reshape(x, (-1, 1))
col_no = 1
row_no = 6

res = gbrt_minimize(
    bench3, dimensions, maxiter=6, n_start=1, random_state=1)
best_xs = res.x_iters.ravel()
best_ys = res.func_vals.ravel()
models = res.models

for n_iter in range(5):
    model = models[n_iter]
    best_x = best_xs[:n_iter+1]
    best_y = best_ys[:n_iter+1]

    low, mu, high = model.predict(vals).T
    std = (high - low) / 2
    acquis_values = -gaussian_ei(vals, model, best_y[-1])
    acquis_values = acquis_values.ravel()
    posterior_mean = mu.ravel()
    posterior_std = std.ravel()
    upper_bound = posterior_mean + posterior_std
    lower_bound = posterior_mean - posterior_std

    plt.subplot(2, 5, col_no)
    plt.plot(x, func_values, color='red', linestyle="--", label="true func")
    plt.plot(x, posterior_mean, color='blue', label="GBRT mean")
    plt.fill_between(
        x, lower_bound, upper_bound, alpha=0.3, color='blue', label="GBRT std")

    sampled_y = [bench3(x) for x in best_x]
    plt.plot(best_x, sampled_y, 'ro', label="observations", markersize=5)
    plt.title("n_iter = %d" % (n_iter + 1))
    plt.ylim([-1.5, 1.5])

    if col_no == 1:
        plt.legend(loc="best", prop={'size': 6}, numpoints=1)
    col_no += 1

    plt.subplot(2, 5, row_no)
    plt.fill_between(x, -1.5, acquis_values, alpha=0.3, color='green',
                     label="EI values")

    min_x = best_xs[n_iter+1]
    plt.plot(min_x, -gaussian_ei(min_x, model, best_y[-1]),
             "ro", markersize=5, label="Next sample point")
    plt.ylim([-1.5, 1.0])

    if row_no == 6:
        plt.legend(loc="best", prop={'size': 8}, numpoints=1)
    row_no += 1

plt.suptitle("Gradient boosted tree based minimization")
plt.show()
