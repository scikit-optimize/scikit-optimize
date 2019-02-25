from functools import partial
from itertools import product

import pytest

from ProcessOptimizer import gp_minimize
from ProcessOptimizer import forest_minimize
from ProcessOptimizer import gbrt_minimize
from ProcessOptimizer import Optimizer
from ProcessOptimizer.learning import ExtraTreesRegressor


# dummy_minimize does not support same parameters so
# treated separately
MINIMIZERS = [gp_minimize]
ACQUISITION = ["LCB", "PI", "EI"]


for est, acq in product(["ET", "RF"], ACQUISITION):
    MINIMIZERS.append(
        partial(forest_minimize, base_estimator=est, acq_func=acq))
for acq in ACQUISITION:
    MINIMIZERS.append(partial(gbrt_minimize, acq_func=acq))


def test_n_random_starts_Optimizer():
    # n_random_starts got renamed in v0.4
    et = ExtraTreesRegressor(random_state=2)
    with pytest.deprecated_call():
        Optimizer([(0, 1.)], et, n_random_starts=10, acq_optimizer='sampling')
