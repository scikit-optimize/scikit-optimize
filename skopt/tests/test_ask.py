from skopt import Optimize
from skopt.benchmarks import branin
from skopt import dummy_minimize


def test_api():
    opt = Optimize(dummy_minimize, [(-5.0, 10.0), (0.0, 15.0)], 1)
    X = opt.suggest()
    y = branin(X)
    opt.report(X, y)
    X2 = opt.suggest()
