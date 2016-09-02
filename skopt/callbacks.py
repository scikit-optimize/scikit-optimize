from collections import Callable
from time import time

import numpy as np


def check_callback(callback):
    """
    Check if callback is a callable or a list of callables.
    """
    if callback is not None:
        if isinstance(callback, Callable):
            return [callback]
        elif (isinstance(callback, list) and
              all([isinstance(c, Callable) for c in callback])):
            return callback
        else:
            raise ValueError("callback should be either a callable or "
                             "a list of callables.")
    else:
        return []


class VerboseCallback(object):
    """
    Callback to control the verbosity.

    Parameters
    ----------
    * `n_init` [int, optional]:
        Number of points provided by the user which are yet to be
        evaluated. This is equal to `len(x0)` when `y0` is None

    * `n_random` [int, optional]:
        Number of points randomly chosen.

    * `n_total` [int]:
        Total number of func calls.

    Attributes
    ----------
    * `iter_no`: [int]:
        Number of iterations of the optimization routine.
    """

    def __init__(self, n_total, n_init=0, n_random=0):
        self.n_init = n_init
        self.n_random = n_random
        self.n_total = n_total
        self.iter_no = 1
        self._start_time = time()
        self._print_info(start=True)

    def _print_info(self, start=True):
        iter_no = self.iter_no
        if start:
            status = "started"
            eval_status = "Evaluating function"
            search_status = "Searching for the next optimal point."
        else:
            status = "ended"
            eval_status = "Evaluation done"
            search_status = "Search finished for the next optimal point."

        if iter_no <= self.n_init:
            print("Iteration No: %d %s. %s at provided point."
                  % (iter_no, status, eval_status))
        elif self.n_init < iter_no <= (self.n_random + self.n_init):
            print("Iteration No: %d %s. %s at random point."
                  % (iter_no, status, eval_status))
        else:
            print("Iteration No: %d %s. %s"
                  % (iter_no, status, search_status))

    def __call__(self, res):
        """
        Parameters
        ----------
        * `res` [`OptimizeResult`, scipy object]:
            The optimization as a OptimizeResult object.
        """
        time_taken = time() - self._start_time
        self._print_info(start=False)

        curr_y = res.func_vals[-1]
        curr_min = res.fun
        print("Time taken: %0.4f" % time_taken)
        print("Function value obtained: %0.4f" % curr_y)
        print("Current minimum: %0.4f" % curr_min)
        self.iter_no += 1
        if self.iter_no <= self.n_total:
            self._print_info(start=True)
            self._start_time = time()


class TimerCallback(object):
    """
    A timer callback that logs the time for every iteration.
    """
    def __init__(self):
        self._time = time()
        self.iter_time = []

    def __call__(self, res):
        """
        Parameters
        ----------
        * `res` [`OptimizeResult`, scipy object]:
            The optimization as a OptimizeResult object.
        """
        elapsed_time = time() - self._time
        self.iter_time.append(elapsed_time)
        self._time = time()
