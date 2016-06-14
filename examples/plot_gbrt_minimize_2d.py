import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from skopt import gbrt_minimize
from skopt.benchmarks import branin
from skopt.acquisition import gaussian_ei

plt.figure(figsize=(10, 10))
plt.set_cmap("viridis")
dimensions = [(-5.0, 10.0), (0.0, 15.0)]

x1_values = np.linspace(-5, 10, 100)
x2_values = np.linspace(0, 15, 100)
x_ax, y_ax = np.meshgrid(x1_values, x2_values)
vals = np.c_[x_ax.ravel(), y_ax.ravel()]

res = gbrt_minimize(
    branin, dimensions, maxiter=200, random_state=1)

model = res.models[-1]
opt_points = res.x_iters
y_opt = res.fun
x_opt = res.x

acquis_values = gaussian_ei(vals, model, y_opt)
acquis_values = acquis_values.reshape(100, 100)

branin_vals = np.reshape([branin(val) for val in vals], (100, 100))

plt.subplot(211)
plt.pcolormesh(x_ax, y_ax, acquis_values)
plt.plot(opt_points[:, 0], opt_points[:, 1], 'ro',
         markersize=4, lw=0, label='samples')
plt.plot(x_opt[0], x_opt[1], 'ws', markersize=8, label='best')
plt.colorbar()
plt.legend(loc='best', numpoints=1)
plt.xlabel('X1')
plt.xlim([-5, 10])
plt.ylabel('X2')
plt.ylim([0, 15])
plt.title("Expected improvement after final iteration")

plt.subplot(212)
plt.pcolormesh(x_ax, y_ax, branin_vals,
               norm=LogNorm(vmin=branin_vals.min(), vmax=branin_vals.max()))
plt.colorbar()
plt.plot(opt_points[:, 0], opt_points[:, 1], 'ro',
         markersize=4, lw=0)
plt.plot(x_opt[0], x_opt[1], 'ws', markersize=8)
plt.xlabel('X1')
plt.xlim([-5, 10])
plt.ylabel('X2')
plt.ylim([0, 15])
plt.title("Branin function")

plt.show()
