.. currentmodule:: skopt.utils
.. _utils:

Utility functions
=================

This is a list of public utility functions. Other functions in this module
are meant for internal use.

:func:`use_named_args`
----------------------
This utility function allows it to use objective functions with named arguments::

    >>> # Define the search-space dimensions. They must all have names!
    >>> from skopt.space import Real
    >>> from skopt.utils import use_named_args
    >>> dim1 = Real(name='foo', low=0.0, high=1.0)
    >>> dim2 = Real(name='bar', low=0.0, high=1.0)
    >>> dim3 = Real(name='baz', low=0.0, high=1.0)
    >>>
    >>> # Gather the search-space dimensions in a list.
    >>> dimensions = [dim1, dim2, dim3]
    >>>
    >>> # Define the objective function with named arguments
    >>> # and use this function-decorator to specify the
    >>> # search-space dimensions.
    >>> @use_named_args(dimensions=dimensions)
    ... def my_objective_function(foo, bar, baz):
    ...     return foo ** 2 + bar ** 4 + baz ** 8

:func:`dump`
------------
Store an skopt optimization result into a file.

:func:`load`
------------
Reconstruct a skopt optimization result from a file persisted with :func:`dump`.
