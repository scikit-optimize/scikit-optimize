# Benchmarks

These results are obtained by running our optimizers on the set of benchmark
functions provided in Table 2 of https://ml.informatik.uni-freiburg.de/papers/13-BayesOpt_EmpiricalFoundation.pdf

All optimizers are run with default values and with `n_calls=200`. Runs are
repeated 10 times.

## Branin

To reproduce, run `python bench_branin.py`

| Method | Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
| dummy_minimize | 0.911 +/- 0.294 |0.492 | 27.6 | 14.677 | 4
| gp_minimize | 0.398 +/- 0.000 |0.398 | 33.1 | 5.7 | 27
| forest_minimize| 0.515 +/- 0.15 |0.399 | 163.8 | 33.295 | 83
| gbrt_minimize | 0.580 +/- 0.33 |0.401 | 110.5 | 49.810 | 46



## Hart6

To reproduce, run `python bench_hart6.py`

| Method | Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
| dummy_minimize | -2.374 +/- 0.327 |-2.909 | 76.5 | 37.876 | 28
| gp_minimize | -3.299 +/- 0.048|-3.322 | 57.3 | 5.658 | 47
| forest_minimize | -3.151 +/- 0.050|-3.225 | 145.2 | 38.992 | 78
| gbrt_minimize | -3.098 +/- 0.114 |-3.248 | 157.2 | 43.904 | 57

## ML, Classification

To reproduce, run `python bench_ml.py`

| Method | Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
| dummy_minimize | 1.293 +/- 0.030 | 1.234 | 39.525 | 5.679 | 22.000
| gp_minimize | 1.287 +/- 0.036 | 1.231 | 47.500 | 1.910 | 43.000
| forest_minimize | 1.311 +/- 0.052 | 1.229 | 40.375 | 8.348 | 15.000
| gbrt_minimize | 1.289 +/- 0.038 | 1.223 | 42.925 | 5.116 | 31.000


## ML, Regression

To reproduce, run `python bench_ml.py`

| Method | Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
| dummy_minimize | -0.769 +/- 0.026 | -0.814 | 40.675 | 6.346 | 23.000
| gp_minimize | -0.775 +/- 0.015 | -0.793 | 41.700 | 6.724 | 25.000
| forest_minimize | -0.771 +/- 0.020 | -0.811 | 35.425 | 9.892 | 15.000
| gbrt_minimize | -0.771 +/- 0.024 | -0.828 | 39.325 | 7.202 | 24.000
