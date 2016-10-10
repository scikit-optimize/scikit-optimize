import numpy as np
from sklearn.gaussian_process.kernels import RBF as sk_RBF

class RBF(sk_RBF):
    def gradient_X(self, X, Y):
        """
        Computes gradient of K(X, Y) with respect to X
        """
        # TODO: Do in-place operations
        length_scale = np.array(self.length_scale)
        diff = X - Y
        diff /= length_scale
        squared = np.exp(-0.5 * np.sum(diff**2, axis=1))
        squared = np.expand_dims(squared, axis=1)
        return -squared * (diff / length_scale)
