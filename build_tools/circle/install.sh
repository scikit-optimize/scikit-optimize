# Adapted shamelessly from https://github.com/scikit-learn-contrib/project-template/blob/master/ci_scripts/install.sh
# Deactivate the circleci-provided virtual environment and setup a
# conda-based environment instead
if [[ `type -t deactivate` ]]; then
  deactivate
fi

# Use the miniconda installer for faster download / install of conda
# itself
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
   numpy scipy cython matplotlib
source activate testenv
pip install git+http://github.com/scikit-learn/scikit-learn.git

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python setup.py develop
