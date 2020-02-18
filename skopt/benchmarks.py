# -*- coding: utf-8 -*-
"""A collection of benchmark problems."""

import numpy as np
from sklearn.utils import check_random_state
from abc import ABCMeta, abstractmethod
import warnings


def bench1(x):
    """A benchmark function for test purposes.

        f(x) = x ** 2

    It has a single minima with f(x*) = 0 at x* = 0.
    .. deprecated::
       bench1 will be removed in 0.8
    """
    warnings.warn("bench1 will be removed in 0.8",
                  DeprecationWarning)
    return x[0] ** 2


def bench1_with_time(x):
    """Same as bench1 but returns the computation time (constant).
    .. deprecated::
       bench1 will be removed in 0.8
    """
    warnings.warn("bench1_with_time will be removed in 0.8",
                  DeprecationWarning)
    return x[0] ** 2, 2.22


def bench2(x):
    """A benchmark function for test purposes.

        f(x) = x ** 2           if x < 0
               (x-5) ** 2 - 5   otherwise.

    It has a global minima with f(x*) = -5 at x* = 5.
    .. deprecated::
       bench1 will be removed in 0.8
    """
    warnings.warn("bench2 will be removed in 0.8",
                  DeprecationWarning)
    if x[0] < 0:
        return x[0] ** 2
    else:
        return (x[0] - 5) ** 2 - 5


def bench3(x):
    """A benchmark function for test purposes.

        f(x) = sin(5*x) * (1 - tanh(x ** 2))

    It has a global minima with f(x*) ~= -0.9 at x* ~= -0.3.
    .. deprecated::
       bench1 will be removed in 0.8
    """
    warnings.warn("bench3 will be removed in 0.8",
                  DeprecationWarning)
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2))


def bench4(x):
    """A benchmark function for test purposes.

        f(x) = float(x) ** 2

    where x is a string. It has a single minima with f(x*) = 0 at x* = "0".
    This benchmark is used for checking support of categorical variables.
    .. deprecated::
       bench1 will be removed in 0.8
    """
    warnings.warn("bench4 will be removed in 0.8",
                  DeprecationWarning)
    return float(x[0]) ** 2


def bench5(x):
    """A benchmark function for test purposes.

        f(x) = float(x[0]) ** 2 + x[1] ** 2

    where x is a string. It has a single minima with f(x) = 0 at x[0] = "0"
    and x[1] = "0"
    This benchmark is used for checking support of mixed spaces.
    .. deprecated::
       bench1 will be removed in 0.8
    """
    warnings.warn("bench5 will be removed in 0.8",
                  DeprecationWarning)
    return float(x[0]) ** 2 + x[1] ** 2


def branin(x, a=1, b=5.1 / (4 * np.pi**2), c=5. / np.pi,
           r=6, s=10, t=1. / (8 * np.pi)):
    """Branin-Hoo function is defined on the square x1 ∈ [-5, 10], x2 ∈ [0, 15].

    It has three minima with f(x*) = 0.397887 at x* = (-pi, 12.275),
    (+pi, 2.275), and (9.42478, 2.475).

    More details: http://www.sfu.ca/~ssurjano/branin.html
    .. deprecated::
       bench1 will be removed in 0.9
    """
    warnings.warn("branin will be removed in 0.9 in favor for Branin",
                  DeprecationWarning)
    return (a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 +
            s * (1 - t) * np.cos(x[0]) + s)


def hart6(x):
    """The six dimensional Hartmann function is defined on the unit hypercube.

    It has six local minima and one global minimum f(x*) = -3.32237 at
    x* = (0.20169, 0.15001, 0.476874, 0.275332, 0.311652, 0.6573).

    More details: http://www.sfu.ca/~ssurjano/hart6.html
    .. deprecated::
       bench1 will be removed in 0.9
    """
    warnings.warn("hart6 will be removed in 0.9 in favor for Hartmann6",
                  DeprecationWarning)
    alpha = np.asarray([1.0, 1.2, 3.0, 3.2])
    P = 10 ** -4 * np.asarray([[1312, 1696, 5569, 124, 8283, 5886],
                               [2329, 4135, 8307, 3736, 1004, 9991],
                               [2348, 1451, 3522, 2883, 3047, 6650],
                               [4047, 8828, 8732, 5743, 1091, 381]])
    A = np.asarray([[10, 3, 17, 3.50, 1.7, 8],
                    [0.05, 10, 17, 0.1, 8, 14],
                    [3, 3.5, 1.7, 10, 17, 8],
                    [17, 8, 0.05, 10, 0.1, 14]])
    return -np.sum(alpha * np.exp(-np.sum(A * (np.array(x) - P)**2, axis=1)))


def hart3(x):
    """The three dimensional Hartmann function is
    defined on the unit hypercube.

    It has four local minima and one global minimum f(x*) = -3.86278 at
    x* = (0.114614, 0.555649, 0.852547).

    More details: http://www.sfu.ca/~ssurjano/hart3.html
    """
    alpha = np.asarray([1.0, 1.2, 3.0, 3.2])
    P = 10 ** -4 * np.asarray([[3689, 1170, 2673],
                               [4699, 4387, 7470],
                               [1091, 8732, 5547],
                               [381, 5743, 8828]])
    A = np.asarray([[3.0, 10, 30],
                    [0.1, 10, 35],
                    [3.0, 10, 30],
                    [0.1, 10, 35]])
    return -np.sum(alpha * np.exp(-np.sum(A * (np.array(x) - P)**2, axis=1)))


def hart4(x):
    """The fourth dimensional Hartmann function is
    defined on the unit hypercube.

    It has four local minima and one global minimum
    f(x*) = -3.6475056362745373 at
    x* = (0.20169, 0.15001, 0.476874, 0.275332).

    More details: http://www.sfu.ca/~ssurjano/hart4.html
    """
    alpha = np.asarray([1.0, 1.2, 3.0, 3.2])
    P = 10 ** -4 * np.asarray([[1312, 1696, 5569, 124],
                               [2329, 4135, 8307, 3736],
                               [2348, 1451, 3522, 2883],
                               [4047, 8828, 8732, 5743]])
    A = np.asarray([[10, 3, 17, 3.50],
                    [0.05, 10, 17, 0.1],
                    [3, 3.5, 1.7, 10],
                    [17, 8, 0.05, 104]])
    return -np.sum(alpha * np.exp(-np.sum(A * (np.array(x) - P)**2, axis=1)))


class BenchmarkFunction(object):
    """Base class Benchmarkfunctions.
    """

    __metaclass__ = ABCMeta

    def __init__(self, n_dim=0, noise_level=0, random_state=None):
        assert n_dim > 0
        assert noise_level >= 0
        self.n_dim = n_dim
        self.noise_level = noise_level
        self.rng = check_random_state(random_state)
        self.min_loc = []
        self.dim = []

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError

    @property
    def minimum_pos(self):
        return self.min_loc

    @property
    def minimum(self):
        return np.asarray([self.__call__(xstar)
                           for xstar in self.minimum_pos])

    @property
    def dimensions(self):
        return self.dim


class Ackley(BenchmarkFunction):
    """Ackley function
    Dimension d, where d is the length of the input vector x.

    The Ackley function is defined on the square xi ∈ [32.768, 32.768]
    for all i =1..d.

    It has many local minima and one global minimum
    f(x*) = 0 at x* = (0., .., 0.).

    More details: http://www.sfu.ca/~ssurjano/ackley.html
    """
    def __init__(self, n_dim=2, noise_level=0, random_state=None):
        super(Ackley, self).__init__(n_dim, noise_level, random_state)
        self.a = 20
        self.b = 0.2
        self.c = 2*np.pi
        self.min_loc = np.array([(0, )*self.n_dim])
        self.dim = [(-32.768, 32.768), ] * self.n_dim

    def __call__(self, x):
        x = np.array(x)
        assert len(x) == self.n_dim
        return -self.a * np.exp(-self.b * np.sqrt(1 / self.n_dim * np.sum(x ** 2))) -\
               np.exp(1 / self.n_dim * np.sum(np.cos(self.c * x))) + self.a + np.exp(1)


class Branin(BenchmarkFunction):
    """Branin-Hoo function is defined on the square x1 ∈ [-5, 10], x2 ∈ [0, 15].

    It has three minima with f(x*) = 0.397887 at x* = (-pi, 12.275),
    (+pi, 2.275), and (9.42478, 2.475).

    More details: http://www.sfu.ca/~ssurjano/branin.html
    """
    def __init__(self, scaled=False, noise_level=0, random_state=None):
        super(Branin, self).__init__(2, noise_level, random_state)
        self.a = 1
        self.b = 5.1 / (4 * np.pi ** 2)
        self.c = 5. / np.pi
        self.r = 6
        self.s = 10
        self.t = 1. / (8 * np.pi)
        self.scaled = scaled
        self.min_loc = np.array([(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)])
        self.dim = [(-5., 10.), (0., 15.)]

    def __call__(self, x):
        x = np.array(x)
        assert len(x) == self.n_dim
        if self.scaled:
            x1 = 15 * x[0] - 5
            x2 = 15 * x[1]
            return 1 / 51.95 * (self.a * (x2 - self.b * x1 ** 2 + self.c * x1 - self.r) ** 2 +
                   self.s * (1 - self.t) * np.cos(x1) + self.s - 44.91) + self.rng.randn() * self.noise_level
        else:
            return (self.a * (x[1] - self.b * x[0] ** 2 + self.c * x[0] - self.r) ** 2 +
                    self.s * (1 - self.t) * np.cos(x[0]) + self.s) + self.rng.randn() * self.noise_level


class Sumquares(BenchmarkFunction):
    """Sumquares function
    Dimension d, where d is the length of the input vector x.

    The Ackley function is defined on the square xi ∈ [-10, 10]
    for all i =1..d.

    It has many local minima and one global minimum
    f(x*) = 0 at x* = (0., .., 0.).

    More details: http://www.sfu.ca/~ssurjano/sumsqu.html
    """
    def __init__(self, n_dim=2, noise_level=0, random_state=None):
        super(Sumquares, self).__init__(n_dim, noise_level, random_state)
        self.min_loc = np.array([(0, )*self.n_dim])
        self.dim = [(-10., 10.), ] * self.n_dim

    def __call__(self, x):
        x = np.array(x)
        assert len(x) == self.n_dim
        return np.sum(np.linspace(1, self.n_dim, self.n_dim) * x**2)


class Sumpow(BenchmarkFunction):
    """Sum of different power functions
    Dimension d, where d is the length of the input vector x.
    :math:`f(x) = \sum_{i=1}^d |x_i|^(i+1)`
    Input space is :math:`x_i \in [-1, 1]` for all :math:`i=1,\ldot,d`.
    Global minimum is :math:`f(x^*)=0, \, x^* = (0,\ldot,0)`
    More details: http://www.sfu.ca/~ssurjano/sumpow.html
    """
    def __init__(self, n_dim=2, noise_level=0, random_state=None):
        super(Sumpow, self).__init__(n_dim, noise_level, random_state)
        self.min_loc = np.array([(0, )*self.n_dim])
        self.dim = [(-10., 10.), ] * self.n_dim

    def __call__(self, x):
        x = np.array(x)
        assert len(x) == self.n_dim
        fx = 0
        x = np.array(x)
        d = len(x)
        for i in range(d):
            fx += np.abs(x[i] ** (i + 2))
        return fx


class Forrester(BenchmarkFunction):
    """Forrester function
    Dimension 1.

    The Forrester function is defined on the xi ∈ [0, 1].

    It has many local minima and one global minimum
    f(x*) = 0 at x* = (0.7572560977886071).

    More details: http://www.sfu.ca/~ssurjano/forretal08.html
    """
    def __init__(self, noise_level=0, random_state=None):
        super(Forrester, self).__init__(1, noise_level, random_state)
        self.min_loc = np.array([[0.7572560977886071]])
        self.dim = [(0., 1.), ]

    def __call__(self, x):
        x = np.array(x)
        assert len(x) == self.n_dim
        return (6. * x[0] - 2.) ** 2 * np.sin(12. * x[0] - 4.)


class Hartmann6(BenchmarkFunction):
    """The six dimensional Hartmann function is defined on the unit hypercube.

    It has six local minima and one global minimum f(x*) = -3.32237 at
    x* = (0.20169, 0.15001, 0.476874, 0.275332, 0.311652, 0.6573).

    More details: http://www.sfu.ca/~ssurjano/hart6.html
    """
    def __init__(self, noise_level=0, random_state=None):
        super(Hartmann6, self).__init__(6, noise_level, random_state)
        self.alpha = np.asarray([1.0, 1.2, 3.0, 3.2])
        self.P = 10 ** -4 * np.asarray([[1312, 1696, 5569, 124, 8283, 5886],
                                        [2329, 4135, 8307, 3736, 1004, 9991],
                                        [2348, 1451, 3522, 2883, 3047, 6650],
                                        [4047, 8828, 8732, 5743, 1091, 381]])
        self.A = np.asarray([[10, 3, 17, 3.50, 1.7, 8],
                             [0.05, 10, 17, 0.1, 8, 14],
                             [3, 3.5, 1.7, 10, 17, 8],
                             [17, 8, 0.05, 10, 0.1, 14]])
        self.min_loc = np.array([[0.20169, 0.15001, 0.476874, 0.275332, 0.311652, 0.6573]])
        self.dim = [(0., 1.), ] * self.n_dim

    def __call__(self, x):
        x = np.array(x)
        assert len(x) == self.n_dim
        return -np.sum(self.alpha * np.exp(-np.sum(self.A * (x - self.P)**2, axis=1)))
