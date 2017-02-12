import pandas as ps
import os
import urllib2 as ul
from cStringIO import StringIO
import cPickle as pc
from scipy.io.arff import loadarff
import xlrd

NUMERIC_OUTPUT = "Number"
CATEGORY_OUTPUT = "Category"

OUTPUT_COLUMN = "output column"
OUTPUT_TYPE = "output type"
DATASET_READER = "reader"

DATASET_FORMAT = "dataset format"

DATASET_TABLE = "table"

def read_arff(x):
    data = loadarff(x)
    return ps.DataFrame(data[0])

def _download_shuffle(url, file_fmt='csv'):

    readers = {'csv': lambda x: ps.read_csv(x),
               'csv_whitespace': lambda x: ps.read_csv(x, delim_whitespace=True),
               'excel': lambda x: ps.read_excel(x),
               'arff': lambda x: read_arff(x)}


    if not file_fmt in readers:
        raise BaseException('You specified unknown format ' + file_fmt + '. Allowed are: ' + str(readers.keys()))

    response = ul.urlopen(url)
    url_data = response.read()
    sio = StringIO(url_data)

    df = readers[file_fmt](sio)

    # shuffle data to avoid non - stationary distributions of outputs
    df = df.sample(frac=1).reset_index(drop=True)
    return df

datasets = {}

def add_dataset(dct, fnc, fmt, otp, tp):
    dct[fnc.__name__] = {
        DATASET_READER: fnc,
        OUTPUT_COLUMN: otp,
        OUTPUT_TYPE: tp,
        DATASET_FORMAT: fmt
    }

# ------------ datasets here --------------

def Credit_Approval():
    return _download_shuffle("https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data")

add_dataset(datasets, Credit_Approval, DATASET_TABLE, -1, CATEGORY_OUTPUT)

def Heart_Disease():
    return _download_shuffle(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat",
        file_fmt="csv_whitespace"
    )

add_dataset(datasets, Heart_Disease, DATASET_TABLE, -1, CATEGORY_OUTPUT)

def German_Credit_Data():
    return _download_shuffle(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
        file_fmt='csv_whitespace'
    )

add_dataset(datasets, German_Credit_Data, DATASET_TABLE, -1, CATEGORY_OUTPUT)

def Thoracic_Surgery_Data():
    return _download_shuffle(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/00277/ThoraricSurgery.arff",
        file_fmt='arff'
    )

add_dataset(datasets, Thoracic_Surgery_Data, DATASET_TABLE, -1, CATEGORY_OUTPUT)

def Climate_Model_Simulation_Crashes():
    return _download_shuffle(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat",
        file_fmt='csv_whitespace'
    )

add_dataset(datasets, Climate_Model_Simulation_Crashes, DATASET_TABLE, -1, CATEGORY_OUTPUT)

#
# numerical target
#
def Automobile():
    return _download_shuffle(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
        file_fmt='excel'
    )

add_dataset(datasets, Automobile, DATASET_TABLE, 0, NUMERIC_OUTPUT)

def Communities_and_Crime():
    return _download_shuffle("http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data")

add_dataset(datasets, Communities_and_Crime, DATASET_TABLE, -1, NUMERIC_OUTPUT)

def Airfoil_Self_Noise():
    return _download_shuffle(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
        file_fmt='csv_whitespace'
    )

add_dataset(datasets, Airfoil_Self_Noise, DATASET_TABLE, -1, NUMERIC_OUTPUT)

def Concrete_Compressive_Strength():
    return _download_shuffle(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
        file_fmt='excel'
    )

add_dataset(datasets, Concrete_Compressive_Strength, DATASET_TABLE, -1, NUMERIC_OUTPUT)

def Energy_Efficiency():
    csv = _download_shuffle(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
        file_fmt="excel"
    )
    csv.drop(csv.columns[-1], axis=1)
    return csv

add_dataset(datasets, Energy_Efficiency, DATASET_TABLE, -1, NUMERIC_OUTPUT)

# ------------ datasets here --------------

datasets_folder = os.path.expanduser('~')
datasets_folder = os.path.join(datasets_folder, ".skopt_eval_datasets")

if not os.path.exists(datasets_folder):
    os.mkdir(datasets_folder)

def data(name):

    try:

        fname = os.path.join(datasets_folder, name)

        if os.path.exists(fname):
            with open(fname) as f:
                return pc.load(f)

        method = datasets[name][DATASET_READER]
        data = method()

        with open(fname, 'w') as f:
            pc.dump(data, f)

        return data

    except AttributeError:
        raise BaseException(name + ": unknown dataset")