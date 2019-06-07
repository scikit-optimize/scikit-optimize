import numpy as np
import pytest

from ProcessOptimizer.space import constraints
from ProcessOptimizer import space

@pytest.mark.new_test
def test_are_constraints_equal():
    a = [constraints.Hard(0,1.0),
    constraints.Hard(2,5.0),
    constraints.Hard(3,'hello'),
    constraints.Hard(4,'a'),
    constraints.Inclusive(1,[0,1]),
    constraints.Exclusive(2,['a,b,c']),
    constraints.Exclusive(3,[3,4]),
    constraints.Inclusive(4,['a,b,c'])]
    b = [None]
    c = []
    assert constraints.are_constraints_equal(a,a)
    assert constraints.are_constraints_equal(b,b)
    assert constraints.are_constraints_equal(c,c)
    assert not constraints.are_constraints_equal(a,b)
    assert not constraints.are_constraints_equal(a,c)
    assert not constraints.are_constraints_equal(b,c)
    assert not constraints.are_constraints_equal(b,a)
    assert not constraints.are_constraints_equal(c,a)
    assert not constraints.are_constraints_equal(b,a)


@pytest.mark.new_test
def test_draw_hard_constrained_values():
    a = constraints.Hard(0,1)
    b = constraints.Hard(2,'a')
    n_samples = 10
    assert all([s == 1 for s in constraints.draw_hard_constrained_values(a,n_samples)])
    assert all([s == 'a' for s in constraints.draw_hard_constrained_values(b,n_samples)])

@pytest.mark.new_test
def test_rvs_constraints():
    n_samples=1000
    SPACE = space.Space([[0,4],['a','b','c'],['one','two','three'],[-1.0,1.0]])
    constraints_list = [constraints.Hard(0,2),
    constraints.Hard(1,'b'),
    constraints.Hard(2,'three'),
    constraints.Hard(3,0.5)]

    samples = constraints.rvs_constraints(SPACE, constraints_list, n_samples=n_samples, random_state = None)
    for sample in samples:
        assert sample[0] == 2
        assert sample[1] == 'b'
        assert sample[2] == 'three'
        assert sample[3] == 0.5

    constraints_list = [constraints.Inclusive(0,[1,3]),
    constraints.Inclusive(1,['a','b']),
    constraints.Inclusive(2,['one']),
    constraints.Inclusive(3,[0.3,0.7])]

    samples = constraints.rvs_constraints(SPACE, constraints_list, n_samples=n_samples, random_state = None)
    for sample in samples:
        assert sample[0] >=1 and sample[0] <=3
        assert sample[1] in ['a','b']
        assert sample[2] == 'one'
        assert sample[3] >= 0.3 and sample[3] <= 0.7


    constraints_list = [constraints.Exclusive(0,[1,3]),
    constraints.Exclusive(1,['a','b']),
    constraints.Exclusive(2,['one']),
    constraints.Exclusive(3,[0.3,0.7])]

    samples = constraints.rvs_constraints(SPACE, constraints_list, n_samples=n_samples, random_state = None)
    for sample in samples:
        assert sample[0] <1 or sample[0] >3
        assert sample[1] not in ['a','b']
        assert not sample[2] == 'one'
        assert sample[3] < 0.3 or sample[3] > 0.7

@pytest.mark.new_test
def test_check_constraints():
    SPACE = space.Space([[0,4],['a','b','c'],['one','two','three'],[-1.0,1.0]])
    constraints_list = [constraints.Exclusive(0,[1,3]),
    constraints.Exclusive(1,['a','b']),
    constraints.Exclusive(2,['one']),
    constraints.Exclusive(3,[0.3,0.7])]
    constraints.check_constraints(SPACE,constraints_list)