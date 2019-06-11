import numpy as np
import pytest

from ProcessOptimizer.space import constraints
from ProcessOptimizer import space

@pytest.mark.new_test
def test_are_constraints_equal():
    a = [constraints.Hard(0,1.0,'real'),
    constraints.Hard(2,5.0,'real'),
    constraints.Hard(3,'hello','categorical'),
    constraints.Hard(4,'a','categorical'),
    constraints.Inclusive(1,[0,1],'integer'),
    constraints.Exclusive(2,['a,b,c'],'categorical'),
    constraints.Exclusive(3,[3,4],'integer'),
    constraints.Inclusive(4,['a,b,c'],'categorical')]
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
def test_rvs_constraints():
    n_samples=1000
    SPACE = space.Space([[0,4],['a','b','c'],['one','two','three'],[-1.0,1.0]])
    constraints_list = [constraints.Hard(0,2,'integer'),
    constraints.Hard(1,'b','categorical'),
    constraints.Hard(2,'three','categorical'),
    constraints.Hard(3,0.5,'real')]

    samples = constraints.rvs_constraints(SPACE, constraints_list, n_samples=n_samples, random_state = None)
    for sample in samples:
        assert sample[0] == 2
        assert sample[1] == 'b'
        assert sample[2] == 'three'
        assert sample[3] == 0.5

    constraints_list = [constraints.Inclusive(0,[1,3],'integer'),
    constraints.Inclusive(1,['a','b'],'categorical'),
    constraints.Inclusive(2,['one'],'categorical'),
    constraints.Inclusive(3,[0.3,0.7],'real')]

    samples = constraints.rvs_constraints(SPACE, constraints_list, n_samples=n_samples, random_state = None)
    for sample in samples:
        assert sample[0] >=1 and sample[0] <=3
        assert sample[1] in ['a','b']
        assert sample[2] == 'one'
        assert sample[3] >= 0.3 and sample[3] <= 0.7


    constraints_list = [constraints.Exclusive(0,[1,3],'integer'),
    constraints.Exclusive(1,['a','b'],'categorical'),
    constraints.Exclusive(2,['one'],'categorical'),
    constraints.Exclusive(3,[0.3,0.7],'real')]

    samples = constraints.rvs_constraints(SPACE, constraints_list, n_samples=n_samples, random_state = None)
    for sample in samples:
        assert sample[0] <1 or sample[0] >3
        assert sample[1] not in ['a','b']
        assert not sample[2] == 'one'
        assert sample[3] < 0.3 or sample[3] > 0.7

@pytest.mark.new_test
def test_check_constraints():
    SPACE = space.Space([[0,4],['a','b','c'],['one','two','three'],[-1.0,1.0]])
    constraints_list = [constraints.Exclusive(0,[1,3],'integer'),
    constraints.Exclusive(1,['a','b'],'categorical'),
    constraints.Exclusive(2,['one'],'categorical'),
    constraints.Exclusive(3,[0.3,0.7],'real')]
    constraints.check_constraints(SPACE,constraints_list)