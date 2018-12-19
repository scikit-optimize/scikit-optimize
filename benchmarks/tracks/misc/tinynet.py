"""
Micro framework for estimations with tiny
feedforward deep neural networks.
"""

import numpy as np

activations = {
    "gauss": lambda x, b: b.exp(-(x)),
    "inv": lambda x, b: 1.0 / (1.0 + x),
    "linear": lambda x, b: x,
    "logistic": lambda x, b: b.log(1 + b.exp(x)),
    "sigmoid": lambda x, b: 1/(1 + b.exp(-x)),
    "quadratic": lambda x, b: x ** 2,
    "LeReLU": lambda x, b: b.maximum(x, x*0.05),
}


def ffnn_predict(X,W, nn_name="nn", backend=np):
    """makes predictions with nn """
    H = X
    idx = 0

    while (nn_name + "_w_%s"%idx) in W:
        w = W[nn_name + "_w_%s"%idx]
        b = W[nn_name + "_b_%s"%idx]
        a = W[nn_name + "_a_%s"%idx]
        H = backend.dot(H,w) + b
        H = activations[a](H, backend)
        idx += 1

    return H
