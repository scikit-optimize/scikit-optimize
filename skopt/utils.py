import numpy as np

def in2d(arr1, arr2):
    """
    Tests if each row of arr1 is in arr2.

    Returns a boolean array with size equal to tne number of rows in arr1.
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    return np.apply_along_axis(
        lambda x: np.any(np.all(x == arr2, axis=1)),
        axis=1, arr=arr1)
