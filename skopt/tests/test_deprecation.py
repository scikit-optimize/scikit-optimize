from functools import partial
from itertools import product

import pytest

from skopt import gp_minimize
from skopt import forest_minimize
from skopt import gbrt_minimize
from skopt import Optimizer
from skopt.benchmarks import branin
from skopt.learning import ExtraTreesRegressor


# dummy_minimize does not support same parameters so
# treated separately
MINIMIZERS = [gp_minimize]
ACQUISITION = ["LCB", "PI", "EI"]


for est, acq in product(["ET", "RF"], ACQUISITION):
    MINIMIZERS.append(
        partial(forest_minimize, base_estimator=est, acq_func=acq))
for acq in ACQUISITION:
    MINIMIZERS.append(partial(gbrt_minimize, acq_func=acq))


@pytest.mark.parametrize("minimizer", MINIMIZERS)
def test_n_random_starts(minimizer):
    # n_random_starts got renamed in v0.4
    with pytest.deprecated_call():
        minimizer(branin, [(-5.0, 10.0), (0.0, 15.0)],
                  n_random_starts=4,
                  n_calls=5,
                  random_state=1)


def test_n_random_starts_Optimizer():
    # as above but for the Optimizer class
    et = ExtraTreesRegressor(random_state=2)
    with pytest.deprecated_call():
        Optimizer([(0, 1.)], et, n_random_starts=10, acq_optimizer='sampling')
