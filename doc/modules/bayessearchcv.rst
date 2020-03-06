.. currentmodule:: skopt

.. _bayessearchcv:

BayesSearchCV, a GridSearchCV compatible estimator
==================================================

Use ``BayesSearchCV`` as a replacement for scikit-learn's GridSearchCV.

BayesSearchCV implements a "fit" and a "score" method.
It also implements "predict", "predict_proba", "decision_function",
"transform" and "inverse_transform" if they are implemented in the
estimator used.

The parameters of the estimator used to apply these methods are optimized
by cross-validated search over parameter settings.

In contrast to GridSearchCV, not all parameter values are tried out, but
rather a fixed number of parameter settings is sampled from the specified
distributions. The number of parameter settings that are tried is
given by n_iter.

Parameters are presented as a list of :class:`skopt.space.Dimension` objects.