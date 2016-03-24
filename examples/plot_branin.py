from math import pi, cos

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from skopt.gp_opt import acquisition_func, gp_minimize

from skopt.tests.test_gp_opt import branin

fig = plt.figure(figsize=plt.figaspect(2.))
fig.suptitle('Acquistion values with EI and UCB after 200 iterations')
bounds = [[-5, 10], [0, 15]]

for i, acq in enumerate(['UCB', 'EI']):
    res = gp_minimize(
        branin, bounds, search='sampling', maxiter=200, random_state=0,
        acq=acq)
    gp_model = res.models[-1]

    x1_values = np.linspace(-5, 10, 100)
    x2_values = np.linspace(0, 15, 100)
    x_ax, y_ax = np.meshgrid(x1_values, x1_values)
    vals = np.asarray(list(zip(x_ax.ravel(), y_ax.ravel())))

    prev_best = None
    if acq == "EI":
        prev_best = res.fun
    acquis_values = acquisition_func(vals, gp_model, bounds=bounds)
    acquis_values = acquis_values.reshape(100, 100)

    colortuple = ('y', 'b')
    colors = np.empty_like(x_ax, dtype=str)
    for y in range(len(x_ax)):
        for x in range(len(x_ax)):
            colors[x, y] = colortuple[(x + y) % len(colortuple)]

    ax = fig.add_subplot(2, 1, i + 1, projection='3d')
    surf = ax.plot_surface(x_ax, y_ax, acquis_values, rstride=1,
                           cstride=1, facecolors=colors,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('X1')
    ax.set_xlim(-10, 15)
    ax.set_ylabel('X2')
    ax.set_ylim(-5, 20)
    ax.set_zlabel('Acquistion values')
    ax.set_zlim(0, 250)
plt.title("Branin acquistion values after the 200th iteration.")
plt.show()
