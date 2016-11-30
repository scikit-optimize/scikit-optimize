# Adapted shamelessly from https://github.com/scikit-learn-contrib/project-template/blob/master/ci_scripts/install.sh
# Deactivate the circleci-provided virtual environment and setup a
# conda-based environment instead
if [[ `type -t deactivate` ]]; then
  deactivate
fi

# Use the miniconda installer for faster download / install of conda
# itself
# XXX: Most of this is very similar to travis/install.sh. We should
# probably refactor it later.
pushd .
cd
mkdir -p download
cd download
echo "Cached in $HOME/download :"
ls -l
echo
if [[ ! -f miniconda.sh ]]
   then
   wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
       -O miniconda.sh
   fi
chmod +x miniconda.sh && ./miniconda.sh -b
cd ..
export PATH="$HOME/miniconda3/bin:$PATH"
conda update --yes conda
popd

# Configure the conda environment and put it in the path using the
# provided versions
conda create -n testenv --yes python pip nose \
   numpy scipy scikit-learn matplotlib
source activate testenv

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python setup.py develop
export SKOPT_HOME=$(pwd)

conda install --yes jupyter
pip install pdoc==0.3.2 pygments

# importing matplotlib once builds the font caches. This avoids
# having warnings in our example notebooks
python -c "import matplotlib.pyplot as plt"
