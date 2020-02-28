"""
===========================
Partial Dependence Plots 2D
===========================

Hvass-Labs Dec 2017
Holger Nahrstaedt 2020

.. currentmodule:: skopt

Simple example to show the new 2D plots.
"""
print(__doc__)
import numpy as np
from math import exp

from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_histogram, plot_objective_2D, plot_objective
from skopt.utils import point_asdict
np.random.seed(123)
import matplotlib.pyplot as plt
#############################################################################

dim_learning_rate = Real(name='learning_rate', low=1e-6, high=1e-2, prior='log-uniform')
dim_num_dense_layers = Integer(name='num_dense_layers', low=1, high=5)
dim_num_dense_nodes = Integer(name='num_dense_nodes', low=5, high=512)
dim_activation = Categorical(name='activation', categories=['relu', 'sigmoid'])

dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_dense_nodes,
              dim_activation]

default_parameters = [1e-4, 1, 64, 'relu']

def model_fitness(x):
    learning_rate, num_dense_layers, num_dense_nodes, activation = x

    fitness = ((exp(learning_rate) - 1.0) * 1000) ** 2 + \
               (num_dense_layers) ** 2 + \
               (num_dense_nodes/100) ** 2

    fitness *= 1.0 + 0.1 * np.random.rand()

    if activation == 'sigmoid':
        fitness += 10

    return fitness

print(model_fitness(x=default_parameters))

#############################################################################

search_result = gp_minimize(func=model_fitness,
                            dimensions=dimensions,
                            n_calls=30,
                            x0=default_parameters,
                            random_state=123
                            )

print(search_result.x)
print(search_result.fun)

#############################################################################

for fitness, x in sorted(zip(search_result.func_vals, search_result.x_iters)):
    print(fitness, x)

#############################################################################

space = search_result.space

print(search_result.x_iters)

search_space = {name: space[name][1] for name in space.dimension_names}

print(point_asdict(search_space, default_parameters))

#############################################################################
print("Plotting now ...")

_ = plot_histogram(result=search_result, dimension_identifier='learning_rate',
                   bins=20)
plt.show()

#############################################################################
_ = plot_objective_2D(result=search_result,
                      dimension_identifier1='learning_rate',
                      dimension_identifier2='num_dense_nodes')
plt.show()

#############################################################################

_ = plot_objective_2D(result=search_result,
                      dimension_identifier1='num_dense_layers',
                      dimension_identifier2='num_dense_nodes')
plt.show()

#############################################################################

_ = plot_objective(result=search_result,
                   plot_dims=['num_dense_layers',
                              'num_dense_nodes'])
plt.show()
