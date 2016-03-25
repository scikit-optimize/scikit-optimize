import numpy as np
import matplotlib.pyplot as plt

from skopt.gp_opt import acquisition_func, gp_minimize
from skopt.tests.test_gp_opt import branin

fig = plt.gcf()
plt.set_cmap("viridis")
bounds = [[-5, 10], [0, 15]]
subplot_no = 311

x1_values = np.linspace(-5, 10, 100)
x2_values = np.linspace(0, 15, 100)
x_ax, y_ax = np.meshgrid(x1_values, x2_values)
vals = np.asarray(list(zip(x_ax.ravel(), y_ax.ravel())))

for i, acq in enumerate(['UCB', 'EI']):
    res = gp_minimize(
        branin, bounds, search='sampling', maxiter=200, random_state=0,
        acq=acq)
    gp_model = res.models[-1]
    opt_points = res['x_iters']

    prev_best = None
    if acq == "EI":
        prev_best = res.fun
    acquis_values = acquisition_func(
        vals, gp_model, bounds=bounds, prev_best=0.0)
    acquis_values = acquis_values.reshape(100, 100)

    plt.subplot(subplot_no)
    plt.pcolormesh(x_ax, y_ax, acquis_values)
    plt.plot(
        opt_points[:, 0], opt_points[:, 1], 'ro', markersize=2)
    plt.colorbar()
    plt.xlabel('X1')
    plt.xlim([-5, 10])
    plt.ylabel('X2')
    plt.ylim([0, 15])
    plt.title(acq)
    subplot_no += 1

plt.subplot(313)
func_vals = np.reshape([branin(val) for val in vals], (100, 100))
plt.pcolormesh(x_ax, y_ax, func_vals)
plt.colorbar()
plt.xlabel('X1')
plt.xlim([-5, 10])
plt.ylabel('X2')
plt.ylim([0, 15])
plt.title("Branin function values.")

plt.show()
