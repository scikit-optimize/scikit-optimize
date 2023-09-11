"""
load_boston.py

This file is used to replace the deprecated import

from sklearn.datasets import fetch_mldata  # deprecated
from sklearn.datasets import load_boston   # deprecated
"""
import numpy as np
import pandas as pd

data_names = {
     "CRIM": "per capita crime rate by town",
     "ZN": "proportion of residential land zoned for lots over 25,000 sq.ft.",
     "INDUS": "proportion of non-retail business acres per town",
     "CHAS": "Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)",
     "NOX": "nitric oxides concentration (parts per 10 million)",
     "RM": "average number of rooms per dwelling",
     "AGE": "proportion of owner-occupied units built prior to 1940",
     "DIS": "weighted distances to five Boston employment centres",
     "RAD": "index of accessibility to radial highways",
     "TAX": "full-value property-tax rate per $10,000",
     "PTRATIO": "pupil-teacher ratio by town",
     "B": "1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town",
     "LSTAT": "% lower status of the population"
}

target_name = {
     "MEDV": "Median value of owner-occupied homes in $1000's"
}

def load_boston():
    data_url = r"http://lib.stat.cmu.edu/datasets/boston"

    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    return [data, target]
