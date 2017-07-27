import numpy as np
from skopt import gp_minimize
import matplotlib.pyplot as plt


def f(x):
    x1 = x[0]
    x2 = x[1]
    a = 1.
    b = 5.1 / (4.*np.pi**2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1. / (8.*np.pi)
    ret = a*(x2-b*x1**2+c*x1-r)**2+s*(1-t)*np.cos(x1)+s

    return ret + np.random.normal()


# repeat 20 times and call 50 times
repetitions = 10
calls = 50

res_ei = list()
for rep in range(repetitions):
    res_ei.append(gp_minimize(f, [(-5.0, 10.0), (0.0, 15.0)], n_calls=calls, acq_func='EI')['func_vals'])
res_ei = np.average(res_ei, axis=0)

# slighlty better?
res_nei_1 = list()
for rep in range(repetitions):
    res_nei_1.append(gp_minimize(f, [(-5.0, 10.0), (0.0, 15.0)], n_calls=calls, acq_func='noisyEI', noisyEI_N_variants=1)['func_vals'])
res_nei_1 = np.average(res_nei_1, axis=0)

# even better?
res_nei_10 = list()
for rep in range(repetitions):
    res_nei_10.append(gp_minimize(f, [(-5.0, 10.0), (0.0, 15.0)], n_calls=calls, acq_func='noisyEI', noisyEI_N_variants=10)['func_vals'])
res_nei_10 = np.average(res_nei_10, axis=0)


print(res_ei)
print(res_nei_1)
print(res_nei_10)

X = range(calls)
plt.plot(X, res_ei)
plt.plot(X, res_nei_1)
plt.plot(X, res_nei_10)

plt.show()
