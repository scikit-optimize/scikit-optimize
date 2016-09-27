These are the results got by optimizing the particular set of benchmark functions provided in Table 2 of
(http://www.automl.org/papers/13-BayesOpt_EmpiricalFoundation.pdf) with our defualt values. All optimizers are run 10 times with `n_calls=200`.

## gp_minimize

|Blackbox Function| Minimum | Best minimum |
------------------|------------|-----------|
|Branin| 0.419 +/- 0.025 |0.398
|Hart6| -3.003 +/- 0.207|-3.308

## forest_minimize

|Blackbox Function| Minimum | Best minimum |
------------------|------------|-----------|
|Branin| 0.420 +/- 0.028 |0.398
|Hart6| -3.111 +/- 0.050|-3.192

## gbrt_minimize

|Blackbox Function| Minimum | Best minimum |
------------------|------------|-----------|
|Branin| 0.475 +/- 0.112 |0.399
|Hart6| -3.128 +/- 0.035|-3.179
