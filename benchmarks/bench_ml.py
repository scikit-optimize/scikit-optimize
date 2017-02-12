import numpy as np
from copy import deepcopy
from multiprocessing import Pool
from skopt.learning import GaussianProcessRegressor, RandomForestRegressor, ExtraTreesRegressor
from _ml_evaluation import Test_Model_On_Dataset, evaluate_optimizer, visualize_traces
import _ml_datasets as datasets
import json # used to save data
from datetime import datetime


if __name__ == "__main__":

    # parameters that will be turned in command line args
    max_iter = 32
    repetitions = 3
    save_traces = False
    run_parallel = True

    selected_algorithms = [ GaussianProcessRegressor, RandomForestRegressor, ExtraTreesRegressor]
    # For the full list of models see: Test_Model_On_Dataset.models
    selected_models = ['DecisionTreeClassifier', 'DecisionTreeRegressor']
    selected_datasets = datasets.datasets.keys()

    pool = Pool()

    # all the parameter values and objectives collected during execution are stored in dict below
    all_data = {}

    for Model in selected_models:
        all_data[Model] = {}
        for Dataset in selected_datasets:

            Supports, Message = Test_Model_On_Dataset.supports(Model, Dataset)

            if not Supports:
                continue

            all_data[Model][Dataset] = {}
            # enumerate all combinations of algos, models, datasets
            for Algorithm in selected_algorithms:
                print(Algorithm.__name__, Model, Dataset)

                params = [(Algorithm, Model, Dataset, max_iter, np.random.randint(2**30)) for rep in range(repetitions)]

                # run experiment 
                if run_parallel:
                    raw_trace = pool.map(evaluate_optimizer, params)
                else:
                    raw_trace = [evaluate_optimizer(p) for p in params] # used for debugging

                all_data[Model][Dataset][Algorithm.__name__] = deepcopy(raw_trace)

    #dump the recorded objective values as json
    if save_traces:
        with open(datetime.now().strftime("%M_%Y_%d_%h_%m_%s")+'.json', 'w') as f:
            json.dump(all_data, f)

    visualize_traces(all_data)

