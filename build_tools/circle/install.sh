conda create -n testenv --yes python pip pytest nose
source activate testenv

pip install -e '.[plots]'
export SKOPT_HOME=$(pwd)

conda install --yes jupyter
pip install pdoc==0.3.2 pygments sphinx sphinx_rtd_theme nbsphinx

# importing matplotlib once builds the font caches. This avoids
# having warnings in our example notebooks
python -c "import matplotlib.pyplot as plt"
