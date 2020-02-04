"""
========================
Partial Dependence Plots
========================

Sigurd Carlsen Feb 2019
Reformatted by Holger Nahrstaedt 2020

.. currentmodule:: skopt

This notebook serves to showcase the new features that are being added to
the scikit-optimize toolbox.
"""
print(__doc__)
import sys
from skopt.plots import plot_objective
from skopt import forest_minimize
import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt


#############################################################################
# plot_objective
# ==============
# Plot objective now supports optional use of partial dependence as well as
# different methods of defining parameter values for dependency plots

# Here we define a function that we evaluate.
def funny_func(x):
    s = 0
    for i in range(len(x)):
        s += (x[i] * i) ** 2
    return s


#############################################################################

# We run forest_minimize on the function
bounds = [(-1, 1.), ] * 4
n_calls = 150

result = forest_minimize(funny_func, bounds, n_calls=n_calls, base_estimator="ET",
                         random_state=4)

# Here we see an example of using partial dependence. Even when setting
# n_points all the way down to 10 from the default of 40, this method is
# still very slow. This is because partial dependence calculates 250 extra
# predictions for each point on the plots.


_ = plot_objective(result, usepartialdependence=True, n_points=10)

# Here we plot without partial dependence. We see that it is a lot faster.
# Also the values for the other parameters are set to the default "result"
# which is the parameter set of the best observed value so far. In the case
# of funny_func this is close to 0 for all parameters.

_ = plot_objective(result, usepartialdependence=False, n_points=10)

# Here we try with setting the other parameters to something other than
# "result". First we try with "expected_minimum" which is the set of
# parameters that gives the miniumum value of the surogate function,
# using scipys minimum search method.

_ = plot_objective(result, usepartialdependence=False, n_points=10,
                   eval_min_params='expected_minimum')

# "expected_minimum_random" is a naive way of finding the minimum of the
# surogate by only using random sampling:

_ = plot_objective(result, usepartialdependence=False, n_points=10,
                   eval_min_params='expected_minimum_random')

# Lastly we can also define these parameters ourselfs by parsing a list
# as the eval_min_params argument:

_ = plot_objective(result, usepartialdependence=False,
                   n_points=10, eval_min_params=[1, -0.5, 0.5, 0])

# We can also specify how many intial samples are used for the two different
# "expected_minimum" methods. We set it to a low value in the next examples
# to showcase how it affects the minimum for the two methods.

_ = plot_objective(result, usepartialdependence=False, n_points=10,
                   eval_min_params='expected_minimum_random',
                   expected_minimum_samples=10)

#############################################################################

_ = plot_objective(result, usepartialdependence=False, n_points=10,
                   eval_min_params='expected_minimum', expected_minimum_samples=1)
