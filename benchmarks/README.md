# Benchmarks

These results are obtained by running our optimizers on the set of benchmark
functions provided in Table 2 of http://www.automl.org/papers/13-BayesOpt_EmpiricalFoundation.pdf

All optimizers are run with default values and with `n_calls=200`. Runs are
repeated 10 times.

## gp_minimize

|Blackbox Function| Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
|Branin| 0.398 +/- 0.000 |0.398 | 57.4 | 9.254 | 44
|Hart6| -3.281 +/- 0.055|-3.321 | 65.1 | 37.522 | 30

## forest_minimize

|Blackbox Function| Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
|Branin| 0.515 +/- 0.15 |0.399 | 163.8 | 33.295 | 83
|Hart6| -3.151 +/- 0.050|-3.225 | 145.2 | 38.992 | 78

## gbrt_minimize

|Blackbox Function| Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
|Branin| 0.580 +/- 0.33 |0.401 | 110.5 | 49.810 | 46
|Hart6| -3.098 +/- 0.114 |-3.248 | 157.2 | 43.904 | 57
