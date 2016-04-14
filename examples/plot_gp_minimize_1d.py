"""
==================================================================
Plot 1-D acquisition values as a function of number of iterations.
==================================================================
"""
print(__doc__)
import numpy as np
import matplotlib.pyplot as plt

from skopt import gp_minimize
from skopt.gp_opt import acquisition

bounds = [[-5, 5]]

vals = np.reshape(np.linspace(-5, 5, 100), (-1, 1))
subplot_no = 131

for n_iter in [2, 5, 10]:
    res = gp_minimize(
        lambda x: x[0]**2, bounds, search='lbfgs', maxiter=n_iter,
        random_state=0, acq='LCB', n_restarts_optimizer=2, n_start=1)
    gp_model = res.models[-1]
    best_x_l = res.x_iters.ravel()

    posterior_mean, posterior_std = gp_model.predict(vals, return_std=True)
    acquis_values = acquisition(vals, gp_model, method="LCB")
    posterior_mean = posterior_mean.ravel()
    posterior_std = posterior_std.ravel()

    plt.subplot(subplot_no)
    plt.plot(vals.ravel(), posterior_mean, label="Posterior mean")
    plt.plot(vals.ravel(), posterior_std, label="Posterior std")
    plt.plot(vals.ravel(), acquis_values, label="Acquisition values.")
    plt.legend(loc="best")
    plt.title("n_iter = %d" % n_iter)
    subplot_no += 1

plt.show()
