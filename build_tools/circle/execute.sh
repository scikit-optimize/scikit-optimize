source activate testenv
export SKOPT_HOME=$(pwd)

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"



cd ~
cd ${SKOPT_HOME}/doc && sphinx-build -M html ${SKOPT_HOME}/doc ${SKOPT_HOME}/doc/_build # -W --keep-going

for entry in ${SKOPT_HOME}/doc/_build/*
do
  echo "$entry"
done