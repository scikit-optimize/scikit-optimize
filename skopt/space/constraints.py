from sklearn.utils import check_random_state
from sklearn.utils.fixes import sp_version
from .space import Real, Integer, Categorical, Space
import numpy as np
class Constraints:
    def __init__(self, constraints_list,space):
        """Constraints used when sampling for the aqcuisiiton function

        Parameters
        ----------
        * `constraints_list` [list of constraints]:
            A list of constraint objects.

        * `space` [Space]:
            A Space object.
        """

        if not isinstance(space,Space):
            space = Space(space)
        check_constraints(space,constraints_list) # Check that all constraints are valid
        self.space = space
        # Lists that keep track of which dimensions has which constraints 
        self.single = [None for _ in range(space.n_dims)]
        self.inclusive = [[] for _ in range(space.n_dims)]
        self.exclusive = [[] for _ in range(space.n_dims)]
        self.sum = []
        self.conditional = []
        self.constraints_list = constraints_list # A copy of the list of constraints
        # Append constraints to the lists
        for constraint in constraints_list:
            if isinstance(constraint,Single):
               self.single[constraint.dimension] = constraint
            elif isinstance(constraint,Inclusive):
                self.inclusive[constraint.dimension].append(constraint)
            elif isinstance(constraint,Exclusive):
                self.exclusive[constraint.dimension].append(constraint)
            elif isinstance(constraint,Sum):
                self.sum.append(constraint)
            elif isinstance(constraint,Conditional):
                self.conditional.append(constraint)
            else:
                raise TypeError('Constraint type {} is not recognized'.format(type(constraint)))

    def __eq__(self, other):
        if isinstance(other,Constraints):
            return self.constraints_list == other.constraints_list
        else:
            return False

    def rvs(self, n_samples = 1, random_state = None):
        """Draw random samples that all are valid with regards to the constraints.

        The samples are in the original space. They need to be transformed
        before being passed to a model or minimizer by `space.transform()`.

        Parameters
        ----------
        * `n_samples` [int, default=1]:
            Number of samples to be drawn from the space.

        * `random_state` [int, RandomState instance, or None (default)]:
            Set random state to something other than None for reproducible
            results.

        Returns
        -------
        * `points`: [list of lists, shape=(n_points, n_dims)]
           Points sampled from the space.
        """

        rng = check_random_state(random_state)
        rows  = [] # A list of the samples drawn
        n_samples_candidates = 0 # Number of candidates for samples that have been checked

        while len(rows) < n_samples: # We keep sampling until all samples a valid with regard to the constraints
            columns = []
            for i in range(self.space.n_dims): # Iterate through all dimensions
                dim = self.space.dimensions[i]
                if self.single[i]: # If a dimension has a "Single"-type constraint we just sample that value
                    column = np.full(n_samples, self.single[i].value)
                else: # Using the default rvs() for the given dimension
                    if sp_version < (0, 16):
                        column = (dim.rvs(n_samples=n_samples))
                    else:
                        column = (dim.rvs(n_samples=n_samples, random_state=rng))
                columns.append(column)

            # Transpose
            for i in range(n_samples):
                r = []
                for j in range(self.space.n_dims):
                    r.append(columns[j][i])
                if self.validate_sample(r): # For each individual sample we check that it is valid
                    # We only append if it is valid
                    rows.append(r)

            n_samples_candidates += n_samples
            if n_samples_candidates > 100000 and len(rows) < 100:
                # If we have less than a 1/10.000 succes rate on the sampling we throw an error
                raise RuntimeError('Could not find valid samples in constrained space. Please check that the constraints allows for valid samples to be drawn')

        # We draw more samples when needed so we only return n_samples of the samples
        return rows[:n_samples]

    def validate_sample(self,sample):
        """ Validates a sample of parameter values in regards to the constraints.

        Parameters
        ----------
        * `sample` [list]:
            A list of values for each dimension.

        Returns
        -------
        * `is_valid`: [bool]
           Returns True for valid samples and False for non-valid samples
        """
       
        # We iterate through all the dimensions and check the the type of constriants that are applied
        # to a single dimensions, i.e Single, Exlcusive and Inclusive
        for dim in range(len(sample)): # We iterate through all samples which corresponds to number of dimensions.
            
            # Single constraints.
            if self.single[dim]:
                if not self.single[dim].validate_constraint(sample[dim]):
                    return False

            # Inclusive constraints.
            if self.inclusive[dim]: # Check if there is a least one inclusive constraint.
                # We go through all inlcusive constraints for this dimension and if the value is not found to be included in any
                # of the bounds of the inclusive constraints we return false.
                value_is_valid = False
                for constraint in self.inclusive[dim]:
                    if constraint.validate_constraint(sample[dim]):
                        value_is_valid = True
                        break
                if not value_is_valid:
                    return False

            # Exclusive constraints
            for constraint in self.exclusive[dim]:
                # The first time a value is inside of the excluded bounds of the exclusive constraint we return false.
                if not constraint.validate_constraint(sample[dim]):
                    return False

        
        # We iterate through sum constriants
        for constraint in self.sum:
            if not constraint.validate_sample(sample):
                return False
        # We iterate through Conditional constraints
        for constraint in self.conditional:
            if not constraint.validate_sample(sample):
                return False
        # If we we did not find any violaiton of the constraints we return True.
        return True

    def __repr__(self):
        return "Constraints({})".format(self.constraints_list)

class Single:
    def __init__(self, dimension, value, dimension_type):
        """Constraint class of type Single.

        This constraint enforces that all values drawn for the specified dimension equals `value`

        Parameters
        ----------
        * `dimension` [int]:
            The index of the dimension for which constraint should be applied.

        * `value` [int, float or str]:
            The enforced value for the constraint.

        * `dimension_type` [str]:
            The type of dimension. Can be 'integer','real' or 'categorical.
            Should be the same type as the dimension it is applied to.
        """

        # Validating input parameters
        if dimension_type == 'categorical':
            if not (type(value) == int or type(value) == str or type(value) == float):
                raise TypeError('Single categorical constraint must be of type int, float or str. Got {}'.format(type(value)))
        elif dimension_type == 'integer':
            if not type(value) == int:
                raise TypeError('Single integer constraint must be of type int. Got {}'.format(type(value)))
        elif dimension_type == 'real':
            if not type(value) == float:
                raise TypeError('Single real constraint must be of type float. Got {}'.format(type(value)))
        else:
            raise ValueError('`dimension_type` must be a string containing "categorical", "integer" or "real". got {}'.format(dimension_type))
        if not (type(dimension) == int):
            raise TypeError('Constraint dimension must be of type int. Got type {}'.format(type(dimension)))
        if not dimension >=0:
            raise ValueError('Dimension can not be a negative number')

        self.value = value
        self.dimension = dimension
        self.dimension_type = dimension_type
        self.validate_constraint = self._validate_constraint
        self.validate_sample = self._validate_sample

    def _validate_sample(self, sample):
        # Returns True if sample does not violate the constraints.
        return self.validate_constraint(sample[self.dimension])

    def _validate_constraint(self, value):
        # Compares value with constraint.
        if value == self.value:
            return True
        else:
            return False

    def __repr__(self):
        return "Single(dimension={}, value={}, dimension_type={})".format(self.dimension, self.value, self.dimension_type)

    def __eq__(self, other):
        if isinstance(other,Single):
            return self.value == other.value and self.dimension == other.dimension and self.dimension_type == other.dimension_type
        else:
            return False

class Bound_constraint():
    def __init__(self, dimension, bounds, dimension_type):
        """Base class for Exclusive and Inclusive constraints"""

        # Validating input parameters
        if not type(bounds) == tuple and not type(bounds) == list:
            raise TypeError('Bounds should be a tuple or a list. Got {}'.format(type(bounds)))
        if not len(bounds) >1:
            raise ValueError('Bounds should be a tuple or list of length > 1.')
        if not dimension_type == 'categorical':
            if not len(bounds) == 2:
                raise ValueError('Length of bounds must be 2 for non-categorical constraints. Got {}'.format(len(bounds)))
        for bound in bounds:
            if dimension_type == 'integer':
                if not type(bound) == int:
                   raise TypeError('Bounds must be of type int for integer dimension. Got {}'.format(type(bound)))
            elif dimension_type == 'real':
                if not type(bound) == float:
                   raise TypeError('Bounds must be of type float for real dimension. Got {}'.format(type(bound)))
        if not (dimension_type == 'categorical' or dimension_type == 'integer' or dimension_type == 'real'):
            raise ValueError('`dimension_type` must be a string containing "categorical", "integer" or "real". Got {}'.format(dimension_type))
        if not (type(dimension) == int):
            raise TypeError('Constraint dimension must be of type int. Got type {}'.format(type(dimension)))
        if not dimension >=0:
            raise ValueError('Dimension can not be a negative number')

        self.bounds = tuple(bounds) # Convert bounds to a tuple
        self.dimension = dimension
        self.dimension_type = dimension_type

class Inclusive(Bound_constraint):
    def __init__(self, dimension, bounds,dimension_type):
        super().__init__(dimension, bounds, dimension_type) 
        """Constraint class of type Inclusive.

        This constraint enforces that all values drawn for the specified dimension are inside the bounds,
        with the bound values included.
        In case of categorical dimensions the constraint enforces that only values specified in bounds can be drawn.

        Parameters
        ----------
        * `dimension` [int]:
            The index of the dimension for which constraint should be applied.

        * `bounds` [tuple]:
            The bounds for the constraint.
            For 'real' dimensions the tuple must be of length 2 and only consist of integers.

            For 'integer' dimensions the tuple must be of length 2 and only consist of floats.

            For 'categorical' dimensions the tuple must be of length < number of dimensions and lenght > 1.
                The tuple can contain any combination of str, int or float

        * `dimension_type` [str]:
            The type of dimension. Can be 'integer','real' or 'categorical.
            Should be the same type as the dimension it is applied to.
        """

        # We use another validation strategy for categorical dimensions.
        if dimension_type == 'categorical':
            self.validate_constraint = self._validate_constraint_categorical
        else:
            self.validate_constraint = self._validate_constraint_real
        self.validate_sample = self._validate_sample

    def _validate_sample(self, sample):
        # Checks if a sample violates the constraints. Returns True if the dont.
        return self.validate_constraint(sample[self.dimension])

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

    def __repr__(self):
        return "Inclusive(dimension={}, bounds={}, dimension_type={})".format(self.dimension, self.bounds, self.dimension_type)

    def __eq__(self, other):
        if isinstance(other, Inclusive):
            return all([a == b for a, b in zip(self.bounds, other.bounds)]) and self.dimension == other.dimension and self.dimension_type == other.dimension_type
        else:
            return False

class Exclusive(Bound_constraint):
    def __init__(self, dimension, bounds, dimension_type):
        super().__init__(dimension, bounds, dimension_type) 
        """Constraint class of type Inclusive.

        This constraint enforces that all values drawn for the specified dimension are outside the bounds,
        with the bound values excluded.
        In case of categorical dimensions the constraint enforces that values specified in bounds can not be drawn.

        Parameters
        ----------
        * `dimension` [int]:
            The index of the dimension for which constraint should be applied.

        * `bounds` [tuple]:
            The bounds for the constraint.
            For 'real' dimensions the tuple must be of length 2 and only consist of integers.

            For 'integer' dimensions the tuple must be of length 2 and only consist of floats.

            For 'categorical' dimensions the tuple must be of length < number of dimensions and lenght > 1.
                The tuple can contain any combination of str, int or float

        * `dimension_type` [str]:
            The type of dimension. Can be 'integer','real' or 'categorical.
            Should be the same type as the dimension it is applied to.
        """

        # We use another validation strategy for categorical dimensions.
        if dimension_type == 'categorical':
            self.validate_constraint = self._validate_constraint_categorical
        else:
            self.validate_constraint = self._validate_constraint_real
        self.validate_sample = self._validate_sample

    def _validate_sample(self, sample):
        # Returns True if sample does not violate the constraints.
        return self.validate_constraint(sample[self.dimension])

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

    def __repr__(self):
        return "Exclusive(dimension={}, bounds={}, dimension_type={})".format(self.dimension, self.bounds, self.dimension_type)

    def __eq__(self, other):
        if isinstance(other, Exclusive):
            return all([a == b for a, b in zip(self.bounds, other.bounds)]) and self.dimension == other.dimension and self.dimension_type == other.dimension_type
        else:
            return False

class Sum():
    def __init__(self, dimensions, value, less_than = True):
        """Constraint class of type Sum.

        This constraint enforces that the sum of all values drawn for the specified dimensions, is less_than or equal to a value.
        Can only be used with integer or real dimensions.

        Parameters
        ----------
        * `dimensions` [list of ints]:
            A list of integers coresponding to the index of the dimensions that should be summed

        * `value` [float or int]:
            The value for which the sum should be less than or equal to.

        * `less_than` [bool, default=True]:
            If true then the sum should be less than or equal to the value. If False the sum should 
            be greater than or equal to the value.
        """

        if not (type(dimensions) == tuple or type(dimensions) == list):
            raise TypeError('Argument `dimensions` must be of type tuple or list. Got {}'.format(type(dimensions)))
        if not len(dimensions) > 1:
            raise ValueError('Argument `dimensions` must have a lenght of more than 1. Got {}'.format(len(dimensions)))
        if not all(type(dim) == int for dim in dimensions):
            raise TypeError('dimension indices must all be of type int.')
        if not all(dim >= 0 for dim in dimensions):
            raise ValueError('Dimension index must be positive')
        if not (type(value) == int or type(value) == float):
            raise TypeError('Argument `value` must be of type float or int. Got {}'.format(type(value)))
        if not type(less_than) == bool:
            raise TypeError('Argument `less_than` must be of type bool (True or False). Got {}'.format(type(less_than)))

        self.dimensions = tuple(dimensions)
        self.value = value
        self.less_than = less_than

        self.validate_sample = self._validate_sample

    def _validate_sample(self, sample):
        # Returns True if sample does not violate the constraints.
        if self.less_than:
            return np.sum([sample[dim] for dim in self.dimensions])<=self.value
        else:
            return np.sum([sample[dim] for dim in self.dimensions])>=self.value


    def __repr__(self):
        return "Sum(dimensions={}, value={}, less_than={})".format(self.dimensions, self.value, self.less_than)

    def __eq__(self, other):
        if isinstance(other, Sum):
            return all([a == b for a, b in zip(self.dimensions, other.dimensions)]) and self.value == other.value and self.less_than == other.less_than
        else:
            return False

class Conditional():
    def __init__(self, condition, if_true = None, if_false = None):
        ''' Constraint class of type Conditional
        
        This constraint enforces other constraints depending on the 'condition'.
        If condition is satisfied the if_true constraint is applied. If not then the if_false
        constraint is applied.

        Parameters
        ----------
        * `condition` [Constraint]:
            A constraint that defines the condition.
            Condition can not be a Conditional constraint.

        * `if_true` [Constraint or tuple/list of constraints, deault=None]:
            If the constraint from the condition-argument is satisfied then this constraint is applied.

        * `if_false` [Constraint or tuple/list of constraints, deault=None]:
            If the constraint from the condition-argument is not satisfied then this constraint is applied.
        '''

        if isinstance(condition,Conditional):
            raise TypeError('Condition can not be a conditional constraint')
        check_is_constraint(condition)
        if if_true:
            if type(if_true) == tuple or type(if_true) == list:
                # Check all constraints
                for constraint in if_true:
                    check_is_constraint(constraint)
                if_true = tuple(if_true)
            else:
                check_is_constraint(if_true)
                # Convert to tuple if only one constraint has been passed
                if_true = tuple([if_true])
        else:
            # Convert "None" to empty tuple
            if_true = ()
        if if_false:
            if type(if_false) == tuple or type(if_false) == list:
                for constraint in if_false:
                    check_is_constraint(constraint)
                    if_false = tuple(if_false)
            else:
                check_is_constraint(if_false)
                if_false = tuple([if_false])
        else:
            if_false = ()

        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false
        self.validate_sample = self._validate_sample

    def _validate_sample(self,sample):
        # Returns True if sample does not violate the constraints.
        if self.condition.validate_sample(sample):
            # Condition evaluates to true
            if self.if_true:
                for constraint in self.if_true: # Iterate through all constrains in if_true
                    if not constraint.validate_sample(sample):
                        # The first time a constraint fails we return False
                        return False
                # If no constraints failed we return True
                return True
            else: # If there is no constraints we return True
                return True
        else:
            if self.if_false:
                for constraint in self.if_false:
                    if not constraint.validate_sample(sample):
                        return False
                return True
            else:
                return True

    def __repr__(self):
        return "Conditional(condition={}, if_true={}, if_false={})".format(self.condition, self.if_true, self.if_false)

    def __eq__(self, other):
        if isinstance(other, Conditional):
            return self.condition == other.condition and self.if_true == other.if_true and self.if_false == other.if_false
        else:
            return False

def check_constraints(space,constraints):
    """ Checks if list of constraints is valid when compared with the dimensions of space.
        Throws errors otherwise.

        Parameters
        ----------
        * `space` [Space]:
            A Space object.

        * `constraints` [list]:
            A list of constraint objects.
    """

    if not type(constraints) == list:
        raise TypeError('Constraints must be a list of constraints. Got {}'.format(type(constraints)))
    n_dims = space.n_dims

    single_constraints = [False]*n_dims
    for constraint in constraints:

        # Check if constraints are inside bounds of space and if more than one single constraint has been
        # applied to the same dimension
        if isinstance(constraint,Single):
            check_dim_and_space(space,constraint)
            ind_dim = constraint.dimension
            if single_constraints[ind_dim] == True:
                raise IndexError('Can not add more than one Singe-type constraint to dimension {}'.format(ind_dim))
            single_constraints[ind_dim] = True
            space_dim = space.dimensions[constraint.dimension]
            check_value(space_dim,constraint.value)
        elif isinstance(constraint,Inclusive) or isinstance(constraint,Exclusive):
            check_dim_and_space(space,constraint)
            space_dim = space.dimensions[constraint.dimension]
            check_bounds(space_dim,constraint.bounds)
        elif isinstance(constraint,Sum):
            if not all(dim < n_dims for dim in constraint.dimensions):
                raise IndexError('Dimension index exceeds number of dimensions')
            for ind_dim in constraint.dimensions:
                if isinstance(space.dimensions[ind_dim],Categorical):
                    raise ValueError('Sum constraint can not be applid to categorical dimension: {}'.format(space.dimensions[ind_dim]))
        elif isinstance(constraint,Conditional):
            # We run check_constraints on each constraint instance in the conditional constraint.
            # We only check them if they are not None
            if constraint.condition: check_constraints(space,[constraint.condition])
            if constraint.if_true: 
                # Check all constraints in the if_true tuple
                for constraint_if_true in constraint.if_true:
                    check_constraints(space,[constraint_if_true])
            if constraint.if_false:
                for constraint_if_false in constraint.if_false:
                    check_constraints(space,[constraint_if_false])
        else:
            raise TypeError('Constraints must be of type "Single", "Exlusive", "Inclusive", "Sum" or "Conditional". Got {}'.format(type(constraint)))

def check_dim_and_space(space,constraint):
    n_dims = space.n_dims
    ind_dim = constraint.dimension # Index of the dimension
    if not ind_dim < n_dims:
        raise IndexError('Dimension index {} out of range for n_dims = {}'.format(ind_dim,n_dims))
    space_dim = space.dimensions[constraint.dimension]

    # Check if space dimensions types are the same as constraint dimension types
    if isinstance(space_dim,Real):
        if not constraint.dimension_type == 'real':
            raise TypeError('Constraint for real dimension {} must be of dimension_type real. Got {}'.format(ind_dim,constraint.dimension_type))
    elif isinstance(space_dim,Integer):
        if not constraint.dimension_type == 'integer':
            raise TypeError('Constraint for integer dimension {} must be of dimension_type integer. Got {}'.format(ind_dim,constraint.dimension_type))
    elif isinstance(space_dim,Categorical):
        if not constraint.dimension_type == 'categorical':
            raise TypeError('Constraint for categorical dimension {} must be of dimension_type categorical. Got {}'.format(ind_dim,constraint.dimension_type))
    else:
        raise TypeError('Can not find valid dimension for' + str(space_dim))

def check_bounds(dim,bounds):
    """ Checks if bounds are included in the dimension.
        Throws errors otherwise.

        Parameters
        ----------
        * `dim` [Dimension]:
            A Space object.

        * `bounds` [tuple]:
            A tuple of int, float or str depending on the type of dimension.
        """

    # Real or integer dimensions.
    if isinstance(dim,Real) or isinstance(dim,Integer):
        for value in bounds:
            if value < dim.low or value > dim.high:
                raise ValueError('Bounds {} exceeds bounds of space {}'.format(bounds,[dim.low,dim.high]))
    else: # Categorical dimensions.
        for value in bounds:
            if value not in dim.categories:
                raise ValueError('Categorical value {} is not in space with categories {}'.format(value,dim.categories))

def check_value(dim,value):
    """ Checks if value are included in the bounds of the dimension.
        Throws errors otherwise.

        Parameters
        ----------
        * `dim` [Dimension]:
            A Space object.

        * `value` [int, float or str]:
            The type of `value` depends on the type of dimension.
        """

    # Real or integer dimension
    if isinstance(dim,Real) or isinstance(dim,Integer):
        if value < dim.low or value > dim.high:
            raise ValueError('Value {} exceeds bounds of space {}'.format(value,[dim.low,dim.high]))
    else: # Categorical dimension.
        if value not in dim.categories:
            raise ValueError('Categorical value {} is not in space with categoreis {}'.format(value,dim.categories))

def check_is_constraint(constraint):
    ''' Checks if constraint is a valid constraint type. Throws error otherwise.
    Types can be Single, Inclusive, Exclusive, Sum and Conditional'''

    if not (isinstance(constraint, Single) or isinstance(constraint, Inclusive)
        or isinstance(constraint, Exclusive) or isinstance(constraint, Sum)
        or isinstance(constraint, Conditional)):
        raise TypeError('Constraint must be of type Inclusive, Exlusive, Single, Sum or Conditional. Got {}'.format(type(constraint)))


    