from skopt.benchmarks import branin, hart6
from skopt.space import Real


class Branin:
    space = [Real(-5,10), Real(0,15)]

    def __call__(self, x):
        return branin(x)


class Hart6:
    space = [Real(0,1) for _ in range(6)]

    def __call__(self, x):
        return hart6(x)


problems = [Branin, Hart6]