# Benchmarks

These results are obtained by running our optimizers on the set of benchmark
functions provided in Table 2 of http://www.automl.org/papers/13-BayesOpt_EmpiricalFoundation.pdf

All optimizers are run with default values and with `n_calls=200`. Runs are
repeated 10 times.

## gp_minimize

|Blackbox Function| Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
|Branin| 0.398 +/- 0.000 |0.398 | 93.8 | 51.581 | 44
|Hart6| -3.281 +/- 0.055|-3.321 | 65.1 | 37.522 | 30

## forest_minimize

|Blackbox Function| Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
|Branin| 0.420 +/- 0.028 |0.398 | 141 | 59.61 | 13
|Hart6| -3.111 +/- 0.050|-3.192 | 90 | 46.57 | 23

## gbrt_minimize

|Blackbox Function| Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
|Branin| 0.475 +/- 0.112 |0.399 | 106.3 | 48.29 | 18
|Hart6| -3.128 +/- 0.035|-3.179 | 122.1 | 55.05 | 50
