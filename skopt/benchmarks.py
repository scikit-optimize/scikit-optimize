"""A collection of benchmark problems"""

import numpy as np


def branin(x, a=1, b=5.1 / (4 * np.pi**2), c=5. / np.pi,
           r=6, s=10, t=1. / (8 * np.pi)):
    """Branin-Hoo function is defined on the square x1 ∈ [-5, 10], x2 ∈ [0, 15].

    It has three minima with f(x*) = 0.397887 at x* = (-pi, 12.275),
    (+pi, 2.275), and (9.42478, 2.475).

    More details: <http://www.sfu.ca/~ssurjano/branin.html>
    """
    return (a * (x[1] - b * x[0]**2 + c * x[0] - r)**2 +
            s * (1 - t) * np.cos(x[0]) + s)

def hart6(x,
          alpha=np.asarray([1.0, 1.2, 3.0, 3.2]),
          P=10**-4 * np.asarray([[1312, 1696, 5569, 124, 8283, 5886],
                                 [2329, 4135, 8307, 3736, 1004, 9991],
                                 [2348, 1451, 3522, 2883, 3047, 6650],
                                 [4047, 8828, 8732, 5743, 1091, 381]]),
          A=np.asarray([[10, 3, 17, 3.50, 1.7, 8],
                        [0.05, 10, 17, 0.1, 8, 14],
                        [3, 3.5, 1.7, 10, 17, 8],
                        [17, 8, 0.05, 10, 0.1, 14]])):
    """The six dimensional Hartmann function is defined on the unit hypercube.

    It has six local minima and one global minimum f(x*) = -3.32237 at
    x* = (0.20169, 0.15001, 0.476874, 0.275332, 0.311652, 0.6573).

    More details: <http://www.sfu.ca/~ssurjano/hart6.html>
    """
    return -np.sum(alpha * np.exp(-np.sum(A * (x - P)**2, axis=1)))
