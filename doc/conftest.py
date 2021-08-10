import os
from os.path import exists
from os.path import join
import warnings

import numpy as np


def pytest_runtest_setup(item):
    fname = item.fspath.strpath
