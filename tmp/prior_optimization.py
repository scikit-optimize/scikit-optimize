from scipy.optimize import minimize, basinhopping, differential_evolution
import numpy as np

class SkoptProxy():
    def __init__(self, ModelClass, bounds):
        self.ModelClass = ModelClass
        self.X = []
        self.Y = []
        self.C = []
        self.context = None
        self.bounds = bounds

    def sample(self):
        r = np.array([np.random.uniform(a,b) for a, b in self.bounds])
        return r

    def set_context(self, c):
        self.context = c

    def concat_context(self, C, X):
        return np.column_stack([C, X])

    def ask(self):
        X, Y = self.X, self.Y
        i = len(Y)

        if i < 10:
            x = self.sample()
            return x

        X = self.concat_context(self.C, X)

        model = GaussianProcessRegressor(kernel=Matern())
        model = model.fit(X, Y)

        y_opt = np.min(Y)

        def acquisition_function(x):
            x = self.concat_context([self.context], [x])
            exploitation, exploration = model.predict(x, return_std=True)
            result = exploitation[0] - exploration[0]  # ((i%3)/2.0)*
            return result

        # find some random x's with good values
        values = []
        for i in range(128):
            x = self.sample()
            y = acquisition_function(x)
            values.append((x, y))

        values.sort(key=lambda v: v[1])

        best_est = np.inf
        best_par = None

        # select the best random points
        for x0, y0 in values[:5]:

            x = minimize(acquisition_function, x0, bounds=self.bounds).x
            y = acquisition_function(x)

            if y < best_est:
                best_est = y
                best_par = x

        return best_par

    def tell(self, x, y):
        self.X.append(x)
        self.Y.append(y)
        self.C.append(self.context)

from skopt import space
from sklearn.svm import SVR

class MultiTaskOptProb():
    def __init__(self, context_features=False, max_subtasks=6):
        self.context_features = context_features

        self.space = [
            space.Real(-5.0, 5.0),
            space.Real(-5.0, 5.0),
            space.Real(-5.0, 5.0),
        ]

        self.context = [
            space.Categorical(range(max_subtasks))
        ]


        if self.context_features:
            self.context += [
                space.Real(0.0,2.0),
                space.Real(0.0,2.0),
            ]

        self.idx = 0

    def reset(self):

        X = np.random.randn(256, 5)
        w = np.random.rand(5)
        Y = np.dot(X, w)

        pw = np.random.uniform(1.0, 2.0)
        nz = np.random.uniform(0.1, 1.0)

        Y = np.sign(Y) * (np.abs(Y) ** pw)
        Y = Y + np.random.randn(len(Y))*nz

        I = np.random.rand(len(X)) < 0.6
        X, Xv = X[I], X[~I]
        Y, Yv = Y[I], Y[~I]


        self.data = X, Y, Xv, Yv

        # return task specific features
        if self.context_features:
            return [pw, nz]
        else:
            return []

    def step(self, P):
        params = {k:10**p for k,p in zip(['C', 'gamma', 'epsilon'], P)}
        model = SVR(**params)
        X, Y, Xv, Yv = self.data
        model.fit(X, Y)
        score = model.score(Xv, Yv)
        return -score



import json
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor, RandomForestRegressor, ExtraTreesRegressor
import random
from skopt.learning.gaussian_process.kernels import Matern
from multiprocessing import Pool


MAX_SUBPROBLEM_STEPS=4
MAX_SIMILAR_TASKS=5
REPEATS = 8
USE_CONTEXT_FEATURES = False
all_classes = [
    lambda: GaussianProcessRegressor(kernel=Matern()),
    lambda: RandomForestRegressor(),
    lambda: ExtraTreesRegressor(),
]
recalculate = False

results_file = 'results.json'

FROM_SCRATCH = "From scratch"
KNOWLEDGE_TRANSFER = "Knowledge transfer"

results = {}

if recalculate:

    for USE_CONTEXT_FEATURES in [False, True]:
        results[USE_CONTEXT_FEATURES] = {}
        for SurrogateClass in all_classes:

            SKS, KTS = [], []

            def repeat_opt_eval(seed):
                np.random.seed(seed)

                eg = MultiTaskOptProb(
                    context_features=USE_CONTEXT_FEATURES,
                    max_subtasks=MAX_SIMILAR_TASKS
                )

                dims = eg.space
                ctx = eg.context

                all_subtraces = {}

                SC, KT = None, None

                for k in [FROM_SCRATCH, KNOWLEDGE_TRANSFER]:

                    solver = Optimizer(
                        dims,
                        SurrogateClass(),
                        acq_optimizer="sampling",
                        context_dimensions=ctx
                    )

                    task = MultiTaskOptProb(
                        context_features=USE_CONTEXT_FEATURES,
                        max_subtasks=MAX_SIMILAR_TASKS
                    )

                    for similar_task in range(MAX_SIMILAR_TASKS):
                        print("Episode " + str(similar_task))

                        trace = []
                        best_y = np.inf

                        feats = task.reset()
                        context = [similar_task]+feats

                        if k == FROM_SCRATCH:
                            solver = Optimizer(
                                dims,
                                SurrogateClass(),
                                acq_optimizer="sampling"
                            )

                        for subp_idx in range(MAX_SUBPROBLEM_STEPS):
                            P = solver.ask()

                            try:
                                v = task.step(P)
                            except BaseException as ex:
                                v = 1.0
                                print ex

                            if best_y > v:
                                best_y = v
                                best_x = P

                            print("#" + str(subp_idx) + ", "+ str(best_y))

                            try:
                                if k == FROM_SCRATCH:
                                    solver.tell(P, v)
                                else:
                                    solver.tell(P, v, ctx=context, next_ctx=context)
                            except BaseException as ex:
                                print ex


                            trace.append(float(best_y))

                    if k == KNOWLEDGE_TRANSFER:
                        KT = trace
                    else:
                        SC = trace

                return SC, KT

            pool = Pool()
            # p = [repeat_opt_eval(v) for v in range(REPEATS)]
            p = pool.map(repeat_opt_eval, range(REPEATS))

            for SC, KT in p:
                SKS.append(SC)
                KTS.append(KT)

            results[USE_CONTEXT_FEATURES][SurrogateClass().__class__.__name__]  = {
                FROM_SCRATCH:SKS,
                KNOWLEDGE_TRANSFER:KTS,
            }

else:
    results = json.load(open(results_file))

with open(results_file, 'w') as f:
    json.dump(results, f)

colors = ['red', 'blue']
name_maps = {KNOWLEDGE_TRANSFER:KNOWLEDGE_TRANSFER, FROM_SCRATCH:FROM_SCRATCH}

def visualize_results(aresults):
    import matplotlib.pyplot as plt

    w = len(aresults.keys())
    h = len(aresults[aresults.keys()[0]].keys())
    idx = 0

    for TF in aresults.keys():
        for B in aresults[TF].keys():
            idx += 1
            results = aresults[TF][B]
            plt.subplot(w,h,idx)

            for k, c in zip(results.keys(), colors):
                y = np.mean(results[k], axis=0)
                x = range(len(y))
                plt.scatter(x, y, c=c, label=name_maps[k])
                plt.xlabel('Iteration')
                plt.ylabel('Avg. objective')
                plt.title(("Task features and number," if (TF == "true") else "Task number, ")+str(B))

            plt.grid()
            plt.legend()

    plt.show()

visualize_results(results)






