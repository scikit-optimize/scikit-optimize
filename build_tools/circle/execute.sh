export PATH="$HOME/miniconda3/bin:$PATH"
source activate testenv
export SKOPT_HOME=$(pwd)

# Generating documentation
for nb in examples/*ipynb; do
    jupyter nbconvert --ExecutePreprocessor.timeout=1024 --execute "$nb" --to markdown |& tee -a nb_to_md.txt
done

cd ~
mkdir -p ./doc/skopt/notebooks
cp ${SKOPT_HOME}/examples/*md ${HOME}/doc/skopt/notebooks
cp -r ${SKOPT_HOME}/examples/*_files ${HOME}/doc/skopt/notebooks
python ${SKOPT_HOME}/build_tools/circle/make_doc.py --overwrite --html --html-dir ./doc --template-dir ${SKOPT_HOME}/build_tools/circle/templates --notebook-dir ./doc/skopt/notebooks skopt
cp -r ./doc ${CIRCLE_ARTIFACTS}
