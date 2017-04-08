import cPickle as pc
import datetime
import time
from copy import deepcopy

from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor, ExtraTreesRegressor, RandomForestRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt.space import Real
import numpy as np
from multiprocessing import Pool

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from _bench_ml import MLBench
import _bench_ml

dataset = MLBench.datasets.keys()[0]
model = "DecisionTreeRegressor"

problem = MLBench(model, dataset, 1)
space = problem.model_description[_bench_ml.MODEL_PARAMETERS]

dimensions_names = space.keys()
dimensions = [space[k][0] for k in dimensions_names]

def fnc(x):
    mlb = MLBench(model, dataset,1)
    point_dct = {k: v for k, v in zip(dimensions_names, x)}
    y = mlb.evaluate(point_dct)
    return y

pool = Pool()

HISTORY_FILE = 'history_1vs2vs4.bin'
PARALLEL_POINTS_VALUES = [(i, tp, regr) for i in [1,2,4] for tp in ['NONE'] for regr in [ExtraTreesRegressor()]]
FUNCTION_EVALUATIONS = 64
REPEATS = 64
LOAD_CACHED_RESULTS = True

if not LOAD_CACHED_RESULTS:

    # all of them points
    history = {}

    for CONFIG in PARALLEL_POINTS_VALUES:

        PARALLEL_POINTS, FILL_TYPE, SURROGATE = CONFIG

        history[CONFIG] = []

        for rep in range(REPEATS):
            print rep, CONFIG

            local_history = []
            X, Y = [], []


            if PARALLEL_POINTS == 1:
                sequential_optimizer = Optimizer(
                        dimensions = dimensions,
                        base_estimator= SURROGATE,
                        acq_optimizer='sampling'
                    )


            for i in range(FUNCTION_EVALUATIONS):


                if PARALLEL_POINTS == 1:
                    Xp = [sequential_optimizer.ask()]
                else:
                    Xe, Ye = deepcopy((X,Y)) # extended set of points used to adjust the variance
                    Xp = [] # points to evaluate in parallel
                    for i in range(PARALLEL_POINTS):
                        # get the points to evaluate
                        opt = Optimizer(
                            dimensions = dimensions,
                            base_estimator= SURROGATE
                        )
                        opt.tell(Xe, Ye)
                        p = opt.ask()
                        #p = deepcopy(p)

                        Xp.append(p)

                        if FILL_TYPE == "RND":
                            Xe.append(p)
                            Ye.append(np.random.rand()*2-1) # random objective for now
                        elif FILL_TYPE == "MEAN":
                            Xe.append(p)
                            mn = np.mean(Y) if len(Y) > 0 else 0.0
                            Ye.append(mn) # random objective for now

                Yp = pool.map(func=fnc, iterable=Xp)

                X.extend(Xp)
                Y.extend(Yp)

                if PARALLEL_POINTS == 1:
                    sequential_optimizer.tell(Xp[0], Yp[0])

                local_history.append(min(Y))

            history[CONFIG].append(local_history)


    with open(HISTORY_FILE, 'w') as f:
        pc.dump(history, f)

history = pc.load(open(HISTORY_FILE))

def render_history(history):

    import matplotlib.pyplot as plt

    hmean, hstd = {}, {}

    for k, v in history.items():
        hmean[k] = np.mean(history[k], axis=0)
        hstd[k] = np.std(history[k], axis=0)

    def plot_multiple(history, name):
        plt.figure()

        for k, v in history.items():
            plt.scatter(range(len(v)), v, label=name + ", " + str(k), c=np.random.rand(3,1))

        plt.xlabel("Sequential step")
        plt.ylabel("Objective value")
        plt.grid()
        plt.legend()


    plot_multiple(hmean, "Min. objective at step")
    plot_multiple(hstd, "Std of min. objective at step")

    plt.show()

render_history(history)
