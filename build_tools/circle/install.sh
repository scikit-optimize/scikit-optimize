conda update -n base conda
conda create -n testenv --yes python pip pytest nose
source activate testenv

python -m pip install -e '.[plots]'
export SKOPT_HOME=$(pwd)

python -m pip install sphinx sphinx-gallery numpydoc memory_profiler

# importing matplotlib once builds the font caches. This avoids
# having warnings in our example notebooks
python -c "import matplotlib.pyplot as plt"
