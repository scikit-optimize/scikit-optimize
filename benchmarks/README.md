# Benchmarks

A set of benchmarks intended for testing black box optimization algorithms.

Benchmarks consist of set of practical optimization problems or problems
 accepted in the literature. Inspired by
http://www.automl.org/papers/13-BayesOpt_EmpiricalFoundation.pdf

## Results

To reproduce, run `python run_all_tests.py`. Every number shown below is

lower confidence bound < mean < upper confidence bound,

where 95% confidence interval is computed using bootstrapping method.
Results below are for 128 runs with 64 calls budget for every problem.

|Method|Branin|Hart6|Select2Features|Train4LayerNN|
|------|------|-----|---------------|-------------|
forest_minimize|0.856<1.15<1.378|-2.945<-2.898<-2.855|-0.398<-0.39<-0.382|-1.005<-0.998<-0.99
gp_minimize|0.398<0.398<0.398|-3.264<-3.219<-3.183|-0.388<-0.379<-0.37|-1.093<-1.087<-1.082
dummy_minimize|1.126<1.295<1.451|-1.854<-1.779<-1.702|-0.369<-0.362<-0.355|-0.789<-0.773<-0.757

## How to install dependencies for benchmarks? ##

* Run pip install -r requirements.txt

### Contribution guidelines ###

All contributions are welcome! :)

If you want to add a benchmark, consider this:

* It needs to have practical relevance and solving corresponding
optimization problem should clearly be valuable. For example, minimizing
random polynomials of power 3 is unlikely to be a problem encountered in
practice. Optimizing components of some medication to improve patient
recovery rate is.
* It needs to simulate a task, where the objective is unknown or complex,
and is expensive to evaluate.
* It needs to run quickly, so that benchmarking does not take days and
 so that progress can be done quickly. A speed - up for realistic
 optimization problem can be obtained by learning predictive models
 from the data to simulate actual objective function.
* It should be hard to guess [near] global optimal value with small
number of random guesses, that is, the problem should not be "easy".
This can be verified by running the dummy_minimize
procedure for many iterations. If the objective does not improve after
small number of iterations, it implies that optimization task is not
too complex.