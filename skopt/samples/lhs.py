"""
Lhs functions are inspired by
https://github.com/clicumu/pyDOE2/blob/
master/pyDOE2/doe_lhs.py

lhs ese is inspired by
https://github.com/damar-wicaksono/gsa-module/blob/develop/
gsa_module/samples/lhs_opt.py (Damar Wicaksono)
"""
import numpy as np
import math
from sklearn.utils import check_random_state
from scipy import spatial
from ..utils import random_permute_matrix
from .base import InitialPointGenerator


class Lhs(InitialPointGenerator):
    """Latin hypercube sampling

    Parameters
    ----------
    lhs_type : str
        - `classic` - a small random number is added
        - `centered` - points are set uniformly in each interval

    criterion : str or None, default=None
        When set to None, the LHS is not optimized

        - `correlation` : optimized LHS by minimizing the correlation

        - `maximin` : optimized LHS by maximizing the minimal pdist

        - `ratio` : optimized LHS by minimizing the ratio
        `max(pdist) / min(pdist)`

        - `ese` : optimized LHS using Enhanced Stochastic Evolutionary Alg.

    iterations : int
        Defines the number of iterations for optimizing LHS
    """
    def __init__(self, lhs_type="classic", criterion=None, iterations=1000):
        self.lhs_type = lhs_type
        self.criterion = criterion
        self.iterations = iterations

        #  ese optimization parameters
        # the initial threshold
        self.ese_threshold_init = 0
        # the number of candidates
        # in perturbation step
        self.ese_num_exchanges = 0
        # the maximum number of inner iterations
        self.ese_max_inner = 0
        # the 2 parameters used in improve process
        #         (a) the cut-off value to decrease the threshold
        #         (b) the multiplier to decrease or increase the threshold
        self.ese_improving_params = [0.1, 0.8]
        # the 4 parameters used in explore process
        #         (a) the cut-off value of acceptance, start increasing
        #         the threshold
        #         (b) the cut-off value of acceptance, start decreasing
        #         the threshold
        #         (c) the cooling multiplier for the threshold
        #         (d) the warming multiplier for the threshold
        self.ese_exploring_params = [0.1, 0.8, 0.9, 0.7]

    def generate(self, n_dim, n_samples, random_state=None):
        """Creates latin hypercube samples.

        Parameters
        ----------
        n_dim : int
           The number of dimension
        n_samples : int
            The order of the LHS sequence. Defines the number of samples.
        random_state : int, RandomState instance, or None (default)
            Set random state to something other than None for reproducible
            results.

        Returns
        -------
        np.array, shape=(n_dim, n_samples)
            LHS set
        """
        rng = check_random_state(random_state)
        if self.criterion is None:
            x = np.linspace(0, 1, n_samples + 1)
            u = rng.rand(n_samples, n_dim)
            h = np.zeros_like(u)
            if self.lhs_type == "centered":
                for j in range(n_dim):
                    h[:, j] = np.diff(x) / 2.0 + x[:n_samples]
            elif self.lhs_type == "classic":
                for j in range(n_dim):
                    h[:, j] = u[:, j] * np.diff(x) + x[:n_samples]
            return random_permute_matrix(h, random_state=random_state)
        else:
            internal_lhs = Lhs(lhs_type=self.lhs_type, criterion=None)
            h_opt = internal_lhs.generate(n_dim, n_samples,
                                          random_state=random_state)
            if self.criterion == "correlation":
                mincorr = np.inf
                for i in range(self.iterations):
                    # Generate a random LHS
                    h = internal_lhs.generate(n_dim, n_samples,
                                              random_state=random_state)
                    r = np.corrcoef(h.T)
                    if np.max(np.abs(r[r != 1])) < mincorr:
                        mincorr = np.max(np.abs(r - np.eye(r.shape[0])))
                        h_opt = h.copy()

            elif self.criterion == "maxmin":
                maxdist = 0
                # Maximize the minimum distance between points
                for i in range(self.iterations):
                    h = internal_lhs.generate(n_dim, n_samples,
                                              random_state=random_state)
                    d = spatial.distance.pdist(h, 'euclidean')
                    if maxdist < np.min(d):
                        maxdist = np.min(d)
                        h_opt = h.copy()

            elif self.criterion == "ratio":
                minratio = np.inf

                # Maximize the minimum distance between points
                for i in range(self.iterations):
                    h = internal_lhs.generate(n_dim, n_samples,
                                              random_state=random_state)
                    p = spatial.distance.pdist(h, 'euclidean')
                    ratio = np.max(p) / np.min(p)
                    if minratio > ratio:
                        minratio = ratio
                        h_opt = h.copy()
            elif self.criterion == "ese":

                dm_init = internal_lhs.generate(n_dim, n_samples,
                                                random_state=random_state)

                if self.ese_threshold_init <= 0.0:
                    threshold = 0.005 * _w2_discrepancy_fast(dm_init)
                else:
                    threshold = self.ese_threshold_init
                if self.ese_num_exchanges <= 0:  # number of exchanges
                    num_exchanges = _calc_num_candidate(n_samples)
                else:
                    num_exchanges = self.ese_num_exchanges
                # maximum number of inner iterations
                if self.ese_max_inner <= 0:
                    max_inner = _calc_max_inner(n_samples, n_dim)
                else:
                    max_inner = self.ese_max_inner

                dm = dm_init.copy()  # the current design
                # the best value of obj.func. so far
                obj_func_best = _w2_discrepancy_fast(dm)
                # the old value of obj.func.
                obj_func_best_old = _w2_discrepancy_fast(dm)
                flag_explore = False  # improved flag

                best_evol = []  # Keep track the best solution
                try_evol = []  # Keep track the accepted trial solution

                # Begin Outer Iteration
                for outer in range(self.iterations):
                    # Initialization of Inner Iteration
                    n_accepted = 0  # number of accepted trial
                    n_improved = 0  # number of improved trial

                    # Begin Inner Iteration
                    for inner in range(max_inner):
                        obj_func = _w2_discrepancy_fast(dm)
                        # Perturb current design
                        num_dimension = inner % n_dim
                        import itertools

                        # Create pairs of all possible combination
                        pairs = list(itertools.combinations(
                            [_ for _ in range(n_samples)], 2))
                        # Create random choices for the pair of
                        # perturbation, w/o replacement
                        rand_choices = rng.choice(len(pairs), num_exchanges,
                                                  replace=False)
                        # Initialize the search
                        obj_func_current = np.inf
                        dm_current = dm.copy()
                        for i in rand_choices:
                            # Always perturb from the design passed
                            # in argument
                            dm_try = dm.copy()
                            # Do column-wise operation in a given
                            # column 'num_dimension'
                            dm_try[pairs[i][0], num_dimension] = dm[
                                pairs[i][1], num_dimension]
                            dm_try[pairs[i][1], num_dimension] = dm[
                                pairs[i][0], num_dimension]
                            obj_func_try = _w2_discrepancy_fast(dm_try)
                            if obj_func_try < obj_func_current:
                                # Select the best trial from all the
                                # perturbation trials
                                obj_func_current = obj_func_try
                                dm_current = dm_try.copy()

                        obj_func_try = _w2_discrepancy_fast(dm_current)
                        # Check whether solution is acceptable
                        if (obj_func_try - obj_func) <=\
                                threshold * rng.rand():
                            # Accept solution
                            dm = dm_current.copy()
                            n_accepted += 1
                            try_evol.append(obj_func_try)
                            if obj_func_try < obj_func_best:
                                # Best solution found
                                h_opt = dm.copy()
                                obj_func_best = obj_func_try
                                best_evol.append(obj_func_best)
                                n_improved += 1

                    # Accept/Reject as Best Solution for convergence checking
                    if ((obj_func_best_old - obj_func_best) /
                        obj_func_best) > 1e-6:
                        # Improvement found
                        obj_func_best_old = obj_func_best
                        # Reset the explore flag after new best found
                        flag_explore = False
                        flag_imp = True
                    else:
                        # Improvement not found
                        flag_imp = False

                    # Improve vs. Explore Phase and Threshold Update
                    if flag_imp:  # Improve
                        # New best solution found, carry out
                        # improvement process
                        if (float(n_accepted / num_exchanges) >
                            self.ese_improving_params[0]) & \
                                (n_accepted > n_improved):
                            # Lots acceptance but not all of them
                            # is improvement,
                            # reduce threshold, make it harder to
                            # accept a trial
                            threshold *= self.ese_improving_params[1]
                        else:
                            # Few acceptance or all trials are improvement,
                            # increase threshold
                            # make it easier to accept a trial
                            threshold /= self.ese_improving_params[1]
                    # Explore, No new best solution found
                    # during last iteration
                    else:
                        # Exploring process, warming up vs. cooling down
                        if n_accepted < self.ese_exploring_params[0] *\
                                num_exchanges:
                            # Reach below limit, increase threshold
                            # ("warming up")
                            flag_explore = True
                        elif n_accepted > self.ese_exploring_params[1] *\
                                num_exchanges:
                            # Reach above limit, decrease threshold
                            # ("cooling down")
                            flag_explore = False

                        if flag_explore:
                            # Ramp up exploration and below upper limit,
                            # increase threshold
                            threshold /= self.ese_exploring_params[3]
                        elif not flag_explore:
                            # Slow down exploration and above lower limit,
                            # decrease threshold
                            threshold *= self.ese_exploring_params[2]

            return h_opt


def _calc_num_candidate(n):
    """Calculate the number of candidates from perturbing the current design
    Recommended in the article is the maximum number of pair combination
    from a given column divided by a factor of 5.
    It is also recommended that the number of candidates to be evaluated does
    not exceed 50

    Parameters
    ----------
    n : int
        the number of elements to be permuted
    Returns
    -------
    the number of candidates from perturbing the current design
        column-wise
    """
    pairs = math.factorial(n) / math.factorial(n - 2) / math.factorial(2)
    fac = 5  # The factor recommended in the article

    return min(int(pairs / fac), 50)


def _calc_max_inner(n, k):
    """Calculate the maximum number of inner iterations
    :math:`\frac{2 \times n_e \times k}{J}`
    It is recommended that the number of inner iterations does not exceed 100
    Parameters
    ----------
    n : int
        the number of samples in the design
    k : int
        the number of design dimension
    Returns
    -------
    the maximum number of inner iterations/loop
    """
    pairs = math.factorial(n) / math.factorial(n - 2) / math.factorial(2)

    return min(int(2 * pairs * k / _calc_num_candidate(n)), 100)


def _w2_discrepancy_fast(D):
    """The vectorized version of wrap-around L2-discrepancy
    calculation, faster!
    The formula for the Wrap-Around L2-Discrepancy is taken from Eq.5 of (1)
    :math:`WD^2(D) = -(4/3)^K + 1/N^2 \Sigma_{i,j=1}^{N} \
    Pi_{k=1}^K [3/2 - |x_k^1 - x_k^2| * (1 - |x_k^1 - x_k^2|)]`
    The implementation below uses a vector operation of numpy array to
    avoid the
    nested loop in the more straightforward implementation

    Parameters
    ----------
    D : np.array
     the design matrix

    Returns
    -------
    the wrap-around L2-discrepancy
    """

    n = D.shape[0]      # the number of samples
    k = D.shape[1]      # the number of dimension
    delta = [None] * k
    for i in range(k):
        # loop over dimension to calculate the absolute difference
        # between point
        # in a given dimension, note the vectorized operation
        delta[i] = np.abs(D[:, i] - np.reshape(D[:, i], (len(D[:, i]), 1)))

    product = 1.5 - delta[0] * (1 - delta[0])
    for i in range(1, k):
        product *= (1.5 - delta[i] * (1 - delta[i]))

    w2_disc = -1 * (4.0/3.0)**k + 1/n**2 * np.sum(product)

    return w2_disc
