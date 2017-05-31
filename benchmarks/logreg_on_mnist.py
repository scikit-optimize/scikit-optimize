"""
Benchmark suggested in practical bayesian optimisation of machine
learning algorithms.
https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf

Runs logistic regression on mnist with 4 hyperparameters.
1. Learning rate for SGD between (0.0, 1.0)
2. L2 regularisation (0.0, 1.0)
3. Mini-batch size (20, 2000)
4. Number of learning epochs (5, 2000)
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

import tensorflow as tf
tf.set_random_seed(42)

from keras.datasets import mnist
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils import np_utils

from skopt.callbacks import TimerCallback
from skopt.plots import plot_convergence
from skopt import gp_minimize
from skopt import forest_minimize
from skopt import dump

(X_train, y_train), (X_test, y_test) = mnist.load_data()
n_classes = len(np.unique(y_train))

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
n_features = X_train.shape[1]

def log_reg_on_mnist(x):
    lr, l2_reg, batch_size, n_epochs = x
    print(x)
    l2_reg = l2(l=l2_reg)
    sgd = SGD(lr=lr)

    input_ = Input(shape=(n_features,))
    logits = Dense(
        n_classes, activation="softmax",
        kernel_regularizer=l2_reg,
        bias_regularizer=l2_reg)(input_)
    model = Model(inputs=input_, outputs=logits)
    model.compile(
        loss=categorical_crossentropy,
        optimizer=sgd, metrics=["accuracy"]
    )
    model.fit(
        X_train, y_train, batch_size=batch_size, epochs=n_epochs,
        verbose=1)
    score = model.evaluate(X_test, y_test)
    if np.any(np.isnan(score)):
        return 1.0
    print("Test loss ", score[0])
    print("Test accuracy", score[1])
    return 1 - score[1]

bounds = [[10**-9, 10**1, "log-uniform"],
          [0.0, 1.0],
          [20, 2000],
          [5, 2000]]

def run(optimizer, n_calls, random_state):
    if optimizer == "gp":
        opt = gp_minimize
    elif optimizer == "forest":
        opt = forest_minimize

    min_vals = []
    all_vals = []
    all_times = []
    timer = TimerCallback()
    res = opt(
        log_reg_on_mnist, bounds,
        verbose=1, random_state=random_state, callback=timer)
    del res["specs"]

    # Pickle instance for further inspection.
    dump(res, "%s_%d.pkl" % (optimizer, random_state))

    # Pickle time
    time_pkl = open("%s_%d_times.pkl" % (optimizer, random_state), "wb")
    pickle.dump(timer.iter_time, time_pkl)
    time_pkl.close()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax = plot_convergence(res, ax=ax)
    plt.savefig("%d.png" % random_state)

    print(res.fun)
    print(res.func_vals)
    print(timer.iter_time)
    return res.fun, res.func_vals


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--optimizer', nargs="?", default="gp", type=str, help="gp | forest")
    parser.add_argument(
        '--n_calls', nargs="?", default="50", type=int, help="Number of calls.")
    parser.add_argument(
        '--random_state', nargs="?", default="5", type=int, help="Random State")
    args = parser.parse_args()
    fun, func_vals = run(args.optimizer, args.n_calls, args.random_state)
    print(fun)
    print(func_vals)
