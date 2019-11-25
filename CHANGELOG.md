# Release history

## Version 0.4.7

### Bugfixes

Changed bokeh version to 1.4.0

## Version 0.4.6

### Bugfixes

ProcessOptimizer.\_\_version\_\_ shows correct version.
Removed \_version.py as we dont use versioneer anymore.
Version needs to be changed manually in \_\_init\_\_ .py from now on.

## Version 0.4.5

Wrong upload. Please don't use this version

## Version 0.4.4

### New features

- Latin hypercube sampling

### Bugfixes

- Progress is now correctly showed in bokeh.

## Version 0.4.3

### Bugfixes

- Lenght scale bounds and length scales were not transformed properly.

## Version 0.4.2

### New features

- optimizer.update_next() added
- Added option to change length scale bounds
- Added optimizer.get_result()
- Added exploration example notebook
- Added length scale bounds example notebook

## Version 0.4.1

### New features

- Draw upper confidence limit in bokeh.
- Colorbar in bokeh
- Same color mapping button in bokeh

## Version 0.4.0

Merged darnr's scikit-optimize fork into ProcessOptimizer. Here is their changelog:

### New features

- `plot_regret` function for plotting the cumulative regret;
  The purpose of such plot is to access how much an optimizer
  is effective at picking good points.
- `CheckpointSaver` that can be used to save a
  checkpoint after each iteration with skopt.dump
- `Space.from_yaml()`
  to allow for external file to define Space parameters

### Bug fixes

- Fixed numpy broadcasting issues in gaussian_ei, gaussian_pi
- Fixed build with newest scikit-learn
- Use native python types inside BayesSearchCV
- Include fit_params in BayesSearchCV refit

### Maintenance

- Added `versioneer` support, to reduce changes with new version of the `skopt`

### Bug fixes

- Separated `n_points` from `n_jobs` in `BayesSearchCV`.
- Dimensions now support boolean np.arrays.

### Maintenance

- `matplotlib` is now an optional requirement (install with `pip install 'scikit-optimize[plots]'`)

High five!

### New features

- Single element dimension definition, which can be used to
  fix the value of a dimension during optimization.
- `total_iterations` property of `BayesSearchCV` that
  counts total iterations needed to explore all subspaces.
- Add iteration event handler for `BayesSearchCV`, useful
  for early stopping inside `BayesSearchCV` search loop.
- added `utils.use_named_args` decorator to help with unpacking named dimensions
  when calling an objective function.

### Bug fixes

- Removed redundant estimator fitting inside `BayesSearchCV`.
- Fixed the log10 transform for Real dimensions that would lead to values being
  out of bounds.

## Version 0.3.3

### New features

- Added text describing progress in bokeh

### Changes

- Changed plot size in bokeh
- ProcessOptimizer now requires tornado 5.1.1

## Version 0.3.0

### New features

- Added constrained parameters

## Version 0.2.0

### New features

- Interactive bokeh GUI for plotting the objective function

## Version 0.0.2

### New features

- Support for using categorical values when plotting objective.

## Version 0.0.1

### New features

- Support for not using partial dependence when plotting objective.
- Support for choosing the values of other parameters when calculating dependence plots
- Support for choosing other minimum search algorithms for the red lines and dots in objective plots
