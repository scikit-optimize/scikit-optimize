# Benchmarks

These results are obtained by running our optimizers on the set of benchmark
functions provided in Table 2 of http://www.automl.org/papers/13-BayesOpt_EmpiricalFoundation.pdf

All optimizers are run with default values and with `n_calls=200`. Runs are
repeated 10 times.

## Branin

To reproduce, run `python bech_branin.py`

| Method | Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
|dummy_minimize | 0.911 +/- 0.294 |0.492 | 27.6 | 14.677 | 4
|gp_minimize | 0.398 +/- 0.000 |0.398 | 33.1 | 5.7 | 27
|forest_minimize| 0.515 +/- 0.15 |0.399 | 163.8 | 33.295 | 83
|gbrt_minimize | 0.580 +/- 0.33 |0.401 | 110.5 | 49.810 | 46



## Hart6

To reproduce, run `python bech_hart6.py`

| Method | Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
|dummy_minimize| -2.374 +/- 0.327 |-2.909 | 76.5 | 37.876 | 28
|gp_minimize | -3.299 +/- 0.048|-3.322 | 57.3 | 5.658 | 47
|forest_minimize | -3.151 +/- 0.050|-3.225 | 145.2 | 38.992 | 78
|gbrt_minimize| -3.098 +/- 0.114 |-3.248 | 157.2 | 43.904 | 57

## ML, Classification

To reproduce, run `python bech_ml.py`

| Method | Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
|gp_minimize| 1.270 +/- 0.026 | 1.231 | NA | NA | NA
|forest_minimize | 1.270 +/- 0.026 | 1.233 | NA | NA | NA
|dummy_minimize | 1.284 +/- 0.029 | 1.241 | NA | NA | NA
|gbrt_minimize| 1.285 +/- 0.043|1.229 | NA | NA | NA


## ML, Regression

To reproduce, run `python bech_ml.py`

| Method | Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
|dummy_minimize | -0.772 +/- 0.023 | -0.802 | NA | NA | NA
|gp_minimize| -0.775 +/- 0.028| -0.814 | NA | NA | NA
|forest_minimize | -0.769 +/- 0.021 | -0.803 | NA | NA | NA
|gbrt_minimize| -0.781 +/- 0.026 | -0.807 | NA | NA | NA
