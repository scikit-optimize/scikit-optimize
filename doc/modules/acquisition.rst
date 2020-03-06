.. currentmodule:: skopt.acquisition

.. _acquisition:

Acquisition
===========
Function to minimize over the posterior distribution.

:class:`gaussian_lcb`
---------------------
Use the lower confidence bound to estimate the acquisition
values.

The trade-off between exploitation and exploration is left to
be controlled by the user through the parameter ``kappa``.

:class:`gaussian_pi`
--------------------
Use the probability of improvement to calculate the acquisition values.

The conditional probability `P(y=f(x) | x)` form a gaussian with a
certain mean and standard deviation approximated by the model.

The PI condition is derived by computing ``E[u(f(x))]``
where ``u(f(x)) = 1``, if ``f(x) < y_opt`` and ``u(f(x)) = 0``,
if ``f(x) > y_opt``.

This means that the PI condition does not care about how "better" the
predictions are than the previous values, since it gives an equal reward
to all of them.

Note that the value returned by this function should be maximized to
obtain the ``X`` with maximum improvement.


:class:`gaussian_ei`
--------------------
Use the expected improvement to calculate the acquisition values.

The conditional probability `P(y=f(x) | x)` form a gaussian with a certain
mean and standard deviation approximated by the model.

The EI condition is derived by computing ``E[u(f(x))]``
where ``u(f(x)) = 0``, if ``f(x) > y_opt`` and ``u(f(x)) = y_opt - f(x)``,
if ``f(x) < y_opt``.

This solves one of the issues of the PI condition by giving a reward
proportional to the amount of improvement got.

Note that the value returned by this function should be maximized to
obtain the ``X`` with maximum improvement.
