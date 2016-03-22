# Shamelessly copied from https://github.com/scikit-learn-contrib/project-template/blob/master/ci_scripts/install.sh

# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

# Use the miniconda installer for faster download / install of conda
# itself
pushd .
cd
mkdir -p download
cd download
echo "Cached in $HOME/download :"
ls -l
echo


# Configure the conda environment and put it in the path using the
# provided versions


pip3 install scipy==0.16.0

source activate testenv

if [[ "$COVERAGE" == "true" ]]; then
    pip3 install coverage coveralls
fi

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python setup.py develop
