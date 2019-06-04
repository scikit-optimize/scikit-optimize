from sklearn.utils import check_random_state
from sklearn.utils.fixes import sp_version

#Integer, Categorical, Real
import numpy as np
class Constraints:
    def __init__(self, constraints_list,space):
        self.hard_constraints = [None]*space.n_dims
        self.inclusive_bounds = [None]*space.n_dims
        self.exclusive_bounds = [None]*space.n_dims
        self.constraints_list = constraints_list
        for constraint in constraints_list:
            if constraint.type == 'hard':
               self.hard_constraints[constraint.parameter] = constraint
class Hard:
    def __init__(self, parameter, value):
        self.value = value
        self.parameter = parameter
        self.type = 'hard'
def check_constraints(space,constraints):
    """ Checks if list of constraints is valid
    """
    from .space import Integer, Categorical, Real
    assert type(constraints) == list, "Constraints must be a list"
    for i in range(len(constraints)):
        assert isinstance(constraints[i],Hard), "Constraint must be an instance of class Hard but constraint %r was of type %r" % (i,type(constraints[i]))
        parameter = constraints[i].parameter
        if isinstance(space.dimensions[parameter],Real):
            assert type(constraints[i].value) == int or type(constraints[i].value) == float, "Constraint for Real dimension must be of type float or int."
        elif isinstance(space.dimensions[parameter],Integer):
            assert type(constraints[i].value) == int, "Constraint for Integer dimension must be of type int."
        elif isinstance(space.dimensions[parameter],Categorical):
            assert type(constraints[i].value) == str, "Constraint for Categorical dimension must be of type str."
def validate_sample(sample, constraints = None):
    """ Validates a sample of of parameter values in regards to the constraints
    Parameters
    ----------
    * `sample`:
        A list of values from each dimension to be checked
    * `constraints`

    """
    pass

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

    columns = []

    for i in range(space.n_dims):
        dim = space.dimensions[i]
        if constraints.hard_constraints[i]:
            column = draw_hard_constrained_values(constraints.hard_constraints[i],n_samples)
        else:
            if sp_version < (0, 16):
                column = (dim.rvs(n_samples=n_samples))
            else:
                column = (dim.rvs(n_samples=n_samples, random_state=rng))
        columns.append(column)

    # Transpose
    rows = []
    for i in range(n_samples):
        r = []
        for j in range(space.n_dims):
            r.append(columns[j][i])

        rows.append(r)

    return rows


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