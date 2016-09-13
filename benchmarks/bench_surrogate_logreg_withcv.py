# We are a big fan of legacy python.

import cPickle
import os
import tarfile
import urllib

import numpy as np
import Surrogates

from skopt import gp_minimize

model_loc_path = "logreg/cv/models/ENCODED_logreg_cv_all_RandomForest"
dir_path = os.path.dirname(os.path.realpath(__file__))
tar_path = os.path.join(dir_path, "logreg_surrogate.tar.gz")
model_loc_path = os.path.join(dir_path, model_loc_path)
print(model_loc_path)

if not os.path.exists(model_loc_path):
    surrogate_url = "http://www.automl.org/downloads/surrogate/logreg_surrogate.tar.gz"
    sur_ret = urllib.urlretrieve(surrogate_url, tar_path)
    tar = tarfile.open(tar_path)
    tar.extractall(path=dir_path)
    tar.close()
    os.remove(tar_path)

logreg_cv = open(model_loc_path, "rb")
surrogate = cPickle.load(logreg_cv)
logreg_cv.close()

sp = surrogate._sp
space_bounds = []
# Configure space Parameters for skopt.
# First argument is surrogate._param_names is fold
for param_name in surrogate._param_names[1:]:
    param_props = sp[param_name]
    param_bounds = [param_props.lower, param_props.upper]
    if param_name == "lrate":
        param_bounds.append("log-uniform")
    space_bounds.append(param_bounds)

def five_fold_log_reg_cv(x):
    mean_accuracies = []
    for x in range(5):
        mean_accuracies.append(surrogate.predict([x] + input_array))
    return np.mean(mean_accuracies)

res = gp_minimize(five_fold_log_reg_cv, space_bounds)
