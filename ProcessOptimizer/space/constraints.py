from sklearn.utils import check_random_state
from sklearn.utils.fixes import sp_version

import numpy as np
class Constraints:
    def __init__(self, constraints_list,space):
        self.hard = [[] for _ in range(space.n_dims)]
        self.inclusive = [[] for _ in range(space.n_dims)]
        self.exclusive = [[] for _ in range(space.n_dims)]
        self.constraints_list = constraints_list
        for constraint in constraints_list:
            if constraint.type == 'hard':
               self.hard[constraint.dimension].append(constraint)
            if constraint.type == 'inclusive':
                self.inclusive[constraint.dimension].append(constraint)
            if constraint.type == 'exclusive':
                self.exclusive[constraint.dimension].append(constraint)
class Hard:
    def __init__(self, dimension, value):
        self.value = value
        self.dimension = dimension
        self.type = 'hard'
class Inclusive:
    def _validate_constraint_categorical(self, value):
        if value in self.bounds:
            return True
        else:
            return False
    def _validate_constraint_real(self, value):
        if value >= self.bounds[0] and value <= self.bounds[1]:
            return True
        else:
            return False
    def __init__(self, dimension, bounds):
        """
        bounds is a list.
        If dimension is integer or real numbers between bounds[0] and bounds[1] are included.
        If dimension is categorical all values found in bounds are inluded.
        """
        self.bounds = bounds
        self.dimension = dimension
        self.type = 'inclusive'
        if type(bounds) == str:
            self.validate_constraints = self._validate_constraint_categorical
        else:
            self.validate_constraints = self._validate_constraint_real
class Exclusive:
    def _validate_constraint_categorical(self, value):
        if value in self.bounds:
            return False
        else:
            return True
    def _validate_constraint_real(self, value):
        if value >= self.bounds[0] and value <= self.bounds[1]:
            return False
        else:
            return True
    def __init__(self, dimension, bounds):
        """
        bounds is a list.
        If dimension is integer or real numbers between bounds[0] and bounds[1] are excluded.
        If dimension is categorical all values found in bounds are excluded.
        """
        self.bounds = bounds
        self.dimension = dimension
        self.type = 'exclusive'
        if type(bounds) == str:
            self.validate_constraints = self._validate_constraint_categorical
        else:
            self.validate_constraints = self._validate_constraint_real

def check_constraints(space,constraints):
    """ Checks if list of constraints is valid
    """
    from .space import Integer, Categorical, Real
    assert type(constraints) == list, "Constraints must be a list"
    for i in range(len(constraints)):
        
        if isinstance(constraints[i],Hard):
            # Hard constraint
            dimension = constraints[i].dimension
            assert type(dimension) == int, 'dimension must be of type int got %r' % type(dimension)
            assert dimension < space.n_dims, 'Constraint dimension exeeds number of dimensions'
            if isinstance(space.dimensions[dimension],Real):
                assert type(constraints[i].value) == int or type(constraints[i].value) == float, "Constraint for Real dimension must be of type float or int."
            elif isinstance(space.dimensions[dimension],Integer):
                assert type(constraints[i].value) == int, "Constraint for Integer dimension must be of type int."
            elif isinstance(space.dimensions[dimension],Categorical):
                assert type(constraints[i].value) == str, "Constraint for Categorical dimension must be of type str."
                assert constraints[i].value in space.dimensions[dimension].categories, "Constraint value %r is not in categories" % constraints[i].value
        elif isinstance(constraints[i],Inclusive) or isinstance(constraints[i],Exclusive):
            dimension = constraints[i].dimension
            assert type(dimension) == int, 'dimension must be of type int got %r' % type(dimension)
            assert dimension < space.n_dims, 'Constraint dimension exeeds number of dimensions'
            for bound in constraints[i].bounds:
                if isinstance(space.dimensions[dimension],Real):
                    assert type(bound) == int or type(bound) == float, "Constraint for Real dimension must be of type float or int."
                elif isinstance(space.dimensions[dimension],Integer):
                    assert type(bound) == int, "Constraint for Integer dimension must be of type int."
                elif isinstance(space.dimensions[dimension],Categorical):
                    assert type(bound) == str, "Constraint for Categorical dimension must be of type str."
                    assert bound in space.dimensions[dimension].categories, "Constraint value %r is not in categories" % bound
        else:
            raise TypeError('Constraints must be of type "hard", "exlusive" or "inclusive" ')

def validate_sample(sample, constraints):
    """ Validates a sample of of parameter values in regards to the constraints
    Parameters
    ----------
    * `sample`:
        A list of values from each dimension to be checked
    * `constraints`

    """
    for i in range(len(sample)):
        for constraint in constraints.exclusive[i]:
            if not constraint.validate_constraints(sample[i]):
                return False
        for constraint in constraints.inclusive[i]:
            if not constraint.validate_constraints(sample[i]):
                return False
    return True

def populate_sample_with_hard_constraints(sample, constraints = None):
    pass

def draw_hard_constrained_values(constraint,n_samples):
    if constraint.type == 'hard':
        column = np.full(n_samples, constraint.value)
    return column

def rvs_constraints(space, constraints_list, n_samples=1, random_state = None):
    """ This function tries to sample valid parameter values.
    This is done by first drawing samples with "hard" constraints
    i.e x0 = 1. Then it checks for validity and draws samples untill enough valid samples
    has been drawn.
    Later som more population rules c
    """
    check_constraints(space,constraints_list)
    constraints = Constraints(constraints_list,space)
    rng = check_random_state(random_state)
    rows  = []
    while len(rows) < n_samples:
        columns = []
        for i in range(space.n_dims):
            dim = space.dimensions[i]
            if constraints.hard[i]:
                column = draw_hard_constrained_values(constraints.hard[i],n_samples)
            else:
                if sp_version < (0, 16):
                    column = (dim.rvs(n_samples=n_samples))
                else:
                    column = (dim.rvs(n_samples=n_samples, random_state=rng))
            columns.append(column)

        # Transpose
        for i in range(n_samples):
            r = []
            for j in range(space.n_dims):
                r.append(columns[j][i])
            if validate_sample(r,constraints):
                rows.append(r)

    return rows[:n_samples]


    # Draw candidates
   # columns = []
   # if constraints: #Check constraints
      #  n_valid_samples = 0
       # rows = []
       # while len(rows) < n_samples:
        #    # Do a normal random sampling
           # for dim in self.dimensions:
           #     if sp_version < (0, 16):
             #       columns.append(dim.rvs(n_samples=n_samples))
            #    else:
              #      columns.append(dim.rvs(n_samples=n_samples, random_state=rng))
                # Populate with hard constraints
                # HERE

            #for i in range(n_samples):
                # Check validity
              #  r = []
              #  for j in range(self.n_dims):
              #      r.append(columns[j][i])
              #  if constraints.validate_sample(r): #
                #    rows.append(r)