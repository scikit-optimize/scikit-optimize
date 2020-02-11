import os
from os.path import exists
from os.path import join
import warnings

import numpy as np

from skopt import IS_PYPY


def pytest_runtest_setup(item):
    fname = item.fspath.strpath
