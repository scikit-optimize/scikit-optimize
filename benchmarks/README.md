# Benchmarks

These results are obtained by running our optimizers on the set of benchmark
functions provided in Table 2 of http://www.automl.org/papers/13-BayesOpt_EmpiricalFoundation.pdf

All optimizers are run with default values and with `n_calls=200`. Runs are
repeated 10 times.

## gp_minimize

|Blackbox Function| Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
|Branin| 0.398 +/- 0.000 |0.398 | 33.1 | 5.7 | 27
|Hart6| -3.299 +/- 0.048|-3.322 | 57.3 | 5.658 | 47
|ML, Classification| 1.270 +/- 0.026 | 1.231 | NA | NA | NA
|ML, Regression| -0.775 +/- 0.028|-0.814 | NA | NA | NA

## forest_minimize

|Blackbox Function| Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
|Branin| 0.515 +/- 0.15 |0.399 | 163.8 | 33.295 | 83
|Hart6| -3.151 +/- 0.050|-3.225 | 145.2 | 38.992 | 78
|ML, Classification| 1.270 +/- 0.026 | 1.233 | NA | NA | NA
|ML, Regression| -0.769 +/- 0.021 | -0.803 | NA | NA | NA

## gbrt_minimize

|Blackbox Function| Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
|Branin| 0.580 +/- 0.33 |0.401 | 110.5 | 49.810 | 46
|Hart6| -3.098 +/- 0.114 |-3.248 | 157.2 | 43.904 | 57
|ML, Classification| 1.285 +/- 0.043|1.229 | NA | NA | NA
|ML, Regression| -0.781 +/- 0.026 | NA | NA | NA

## dummy_minimize

|Blackbox Function| Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
|ML, Classification| 1.284 +/- 0.029 | 1.241 | NA | NA | NA
|ML, Regression| -0.772 +/- 0.023 | -0.802 | NA | NA | NA
