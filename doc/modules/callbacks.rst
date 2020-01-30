.. currentmodule:: skopt.callbacks

.. _callbacks:

Callbacks
=========

Monitor and influence the optimization procedure via callbacks.

Callbacks are callables which are invoked after each iteration of the optimizer
and are passed the results "so far". Callbacks can monitor progress, or stop
the optimization early by returning `True`.

Monitoring callbacks
--------------------

* VerboseCallback
* TimerCallback

Early stopping callbacks
------------------------

* DeltaXStopper
* DeadlineStopper
* DeltaXStopper
* DeltaYStopper
* EarlyStopper

Other callbacks
---------------

* CheckpointSaver

