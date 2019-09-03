from math import isclose

import numpy as np
def test_npvar():

	vector=np.array([0,1])
	expected=0.25
	result=np.var(vector)
	assert isclose(expected,result)


def test_npvarfuzzy():

	rand_state = np.random.RandomState(1333)

	N = 100000
	x = rand_state.randn(N)

	assert isclose(1., np.var(x), rel_tol=0.05)

    
