"""
==================================
Comparing initial sampling methods
==================================

Holger Nahrstaedt 2020 Sigurd Carlsen October 2019

.. currentmodule:: skopt


When doing baysian optimization we often want to reserve some of the
early part of the optimization to pure exploration. By default the
optimizer suggests purely random samples for the first n_initial_points
(10 by default). The downside to this is that there is no guarantee that
these samples are spread out evenly across all the dimensions.

Sampling methods as Latin hypercube, Sobol, Halton and Hammersly
take advantage of the fact that we know beforehand how many random
points we want to sample. Then these points can be "spread out" in
such a way that each dimension is explored.

See also the example on an integer space
:ref:`sphx_glr_auto_examples_initial_sampling_method_integer.py`
"""

print(__doc__)
import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.samples import Sobol
from skopt.samples import Lhs
from skopt.samples import Halton
from skopt.samples import Hammersly
from scipy.spatial.distance import pdist

#############################################################################

def plot_branin(x, title):
    fig, ax = plt.subplots()
    plt.plot(np.array(x)[:, 0], np.array(x)[:, 1], 'bo', label='samples')
    plt.plot(np.array(x)[:, 0], np.array(x)[:, 1], 'bo', markersize=40, alpha=0.5)
    # ax.legend(loc="best", numpoints=1)
    ax.set_xlabel("X1")
    ax.set_xlim([-5, 10])
    ax.set_ylabel("X2")
    ax.set_ylim([0, 15])
    plt.title(title)

n_dim = 2
n_samples = 40

space = Space([(-5., 10.), (0., 15.)])
space.set_transformer("normalize")

#############################################################################
# Random sampling
# ---------------
x = space.rvs(n_samples)
plot_branin(x, "Random samples")
pdist_data = []
x_label = []
pdist_data.append(pdist(x).flatten())
x_label.append("random")
#############################################################################
# Sobol
# -----

sobol = Sobol()
inv_initial_samples = sobol.generate(n_dim, n_samples)
x = space.inverse_transform(inv_initial_samples)
plot_branin(x, 'Sobol')
pdist_data.append(pdist(x).flatten())
x_label.append("sobol")


#############################################################################
# Classic Latin hypercube sampling
# --------------------------------

lhs = Lhs(lhs_type="classic")
inv_initial_samples = lhs.generate(n_dim, n_samples)
x = space.inverse_transform(inv_initial_samples)
plot_branin(x, 'classic LHS')
pdist_data.append(pdist(x).flatten())
x_label.append("lhs")

#############################################################################
# Centered Latin hypercube sampling
# ---------------------------------

lhs = Lhs(lhs_type="centered")
inv_initial_samples = lhs.generate(n_dim, n_samples)
x = space.inverse_transform(inv_initial_samples)
plot_branin(x, 'centered LHS')
pdist_data.append(pdist(x).flatten())
x_label.append("center")

#############################################################################
# Maximin optimized hypercube sampling
# ------------------------------------

lhs = Lhs(criterion="maximin", iterations=1000)
inv_initial_samples = lhs.generate(n_dim, n_samples)
x = space.inverse_transform(inv_initial_samples)
plot_branin(x, 'maximin LHS')
pdist_data.append(pdist(x).flatten())
x_label.append("maximin")

#############################################################################
# Correlation optimized hypercube sampling
# ----------------------------------------

lhs = Lhs(criterion="correlation", iterations=1000)
inv_initial_samples = lhs.generate(n_dim, n_samples)
x = space.inverse_transform(inv_initial_samples)
plot_branin(x, 'correlation LHS')
pdist_data.append(pdist(x).flatten())
x_label.append("corr")

#############################################################################
# Ratio optimized hypercube sampling
# ----------------------------------

lhs = Lhs(criterion="ratio", iterations=1000)
inv_initial_samples = lhs.generate(n_dim, n_samples)
x = space.inverse_transform(inv_initial_samples)
plot_branin(x, 'ratio LHS')
pdist_data.append(pdist(x).flatten())
x_label.append("ratio")

#############################################################################
# Halton sampling
# ---------------

halton = Halton()
inv_initial_samples = halton.generate(n_dim, n_samples)
x = space.inverse_transform(inv_initial_samples)
plot_branin(x, 'Halton')
pdist_data.append(pdist(x).flatten())
x_label.append("halton")

#############################################################################
# Hammersly sampling
# ------------------

hammersly = Hammersly()
inv_initial_samples = hammersly.generate(n_dim, n_samples)
x = space.inverse_transform(inv_initial_samples)
plot_branin(x, 'Hammersly')
pdist_data.append(pdist(x).flatten())
x_label.append("hammersly")

#############################################################################
# Pdist boxplot of all methods
# ----------------------------
#
# This boxplot shows the distance between all generated points using
# Euclidian distance. The higher the value, the better the sampling method.
# It can be seen that random has the worst performance

fig, ax = plt.subplots()
ax.boxplot(pdist_data)
plt.grid(True)
plt.ylabel("pdist")
_ = ax.set_ylim(0, 12)
_ = ax.set_xticklabels(x_label, rotation=45, fontsize=8)