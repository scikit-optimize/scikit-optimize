"""
===========================================
Store and load `skopt` optimization results
===========================================

Mikhail Pak, October 2016.
Reformatted by Holger Nahrstaedt 2020

.. currentmodule:: skopt

Problem statement
=================

We often want to store optimization results in a file. This can be useful,
for example,

* if you want to share your results with colleagues;
* if you want to archive and/or document your work;
* or if you want to postprocess your results in a different Python instance or on an another computer.

The process of converting an object into a byte stream that can be stored in
a file is called _serialization_.
Conversely, _deserialization_ means loading an object from a byte stream.

**Warning:** Deserialization is not secure against malicious or erroneous
code. Never load serialized data from untrusted or unauthenticated sources!

"""
print(__doc__)
import numpy as np
import os
import sys

# The followings are hacks to allow sphinx-gallery to run the example.
sys.path.insert(0, os.getcwd())
main_dir = os.path.basename(sys.modules['__main__'].__file__)
IS_RUN_WITH_SPHINX_GALLERY = main_dir != os.getcwd()

#############################################################################
# Simple example
# ==============
#
# We will use the same optimization problem as in the
# :ref:`sphx_glr_auto_examples_bayesian-optimization.py` notebook:

from skopt import gp_minimize
noise_level = 0.1

if IS_RUN_WITH_SPHINX_GALLERY:
    # When this example is run with sphinx gallery, it breaks the pickling
    # capacity for multiprocessing backend so we have to modify the way we
    # define our functions. This has nothing to do with the example.
    from utils import obj_fun
else:
    def obj_fun(x, noise_level=noise_level):
        return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) + np.random.randn() * noise_level

res = gp_minimize(obj_fun,            # the function to minimize
                  [(-2.0, 2.0)],      # the bounds on each dimension of x
                  x0=[0.],            # the starting point
                  acq_func="LCB",     # the acquisition function (optional)
                  n_calls=15,         # the number of evaluations of f including at x0
                  n_random_starts=0,  # the number of random initialization points
                  random_state=777)

#############################################################################
# As long as your Python session is active, you can access all the
# optimization results via the `res` object.
#
# So how can you store this data in a file? `skopt` conveniently provides
# functions :class:`skopt.dump` and :class:`skopt.load` that handle this for you.
# These functions are essentially thin wrappers around the
# `joblib <https://joblib.readthedocs.io/en/latest/>`_ module's :obj:`joblib.dump` and :obj:`joblib.load`.
#
# We will now show how to use :class:`skopt.dump` and :class:`skopt.load` for storing
# and loading results.
#
# Using `skopt.dump()` and `skopt.load()`
# =======================================
#
# For storing optimization results into a file, call the :class:`skopt.dump`
# function:

from skopt import dump, load

dump(res, 'result.pkl')

#############################################################################
# And load from file using :class:`skopt.load`:

res_loaded = load('result.pkl')

res_loaded.fun

#############################################################################
# You can fine-tune the serialization and deserialization process by calling
# :class:`skopt.dump` and :class:`skopt.load` with additional keyword arguments. See the
# `joblib <https://joblib.readthedocs.io/en/latest/>`_ documentation
# :obj:`joblib.dump` and
# :obj:`joblib.load` for the additional parameters.
#
# For instance, you can specify the compression algorithm and compression
# level (highest in this case):

dump(res, 'result.gz', compress=9)

from os.path import getsize
print('Without compression: {} bytes'.format(getsize('result.pkl')))
print('Compressed with gz:  {} bytes'.format(getsize('result.gz')))

#############################################################################
# Unserializable objective functions
# ----------------------------------
#
# Notice that if your objective function is non-trivial (e.g. it calls MATLAB
# engine from Python), it might be not serializable and :class:`skopt.dump` will
# raise an exception when you try to store the optimization results.
# In this case you should disable storing the objective function by calling
# :class:`skopt.dump` with the keyword argument `store_objective=False`:

dump(res, 'result_without_objective.pkl', store_objective=False)

#############################################################################
# Notice that the entry `'func'` is absent in the loaded object but is still
# present in the local variable:


res_loaded_without_objective = load('result_without_objective.pkl')

print('Loaded object: ', res_loaded_without_objective.specs['args'].keys())
print('Local variable:', res.specs['args'].keys())

#############################################################################
# Possible problems
# =================
#
# * **Python versions incompatibility:** In general, objects serialized in
#   Python 2 cannot be deserialized in Python 3 and vice versa.
# * **Security issues:** Once again, do not load any files from untrusted
#   sources.
# * **Extremely large results objects:** If your optimization results object
#
# is extremely large, calling :class:`skopt.dump` with `store_objective=False` might
# cause performance issues. This is due to creation of a deep copy without the
# objective function. If the objective function it is not critical to you, you
# can simply delete it before calling :class:`skopt.dump`. In this case, no deep
# copy is created:

del res.specs['args']['func']

dump(res, 'result_without_objective_2.pkl')
