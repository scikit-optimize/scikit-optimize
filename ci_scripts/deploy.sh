#!/bin/bash
echo "Running deployment script..."

conda install --yes jupyter
pip install pdoc==0.3.2 pygments


# Generating documentation
cd ~
mkdir -p ./doc/skopt/notebooks
cd ./doc/skopt/notebooks

OIFS="$IFS"
IFS=$'\n'
for nb in ${TRAVIS_BUILD_DIR}/examples/*ipynb; do
    jupyter nbconvert --execute "$nb" --to markdown
done
cp ${TRAVIS_BUILD_DIR}/examples/*md .
cp -r ${TRAVIS_BUILD_DIR}/examples/*_files .
IFS="$OIFS"

cd ~
python ${TRAVIS_BUILD_DIR}/ci_scripts/make_doc.py --overwrite --html --html-dir ./doc --template-dir ${TRAVIS_BUILD_DIR}/ci_scripts/templates --notebook-dir ./doc/skopt/notebooks skopt

# Copying to github pages
echo "Copying built files"
git clone -b master "https://${GH_TOKEN}@github.com/scikit-optimize/scikit-optimize.github.io.git" deploy > /dev/null 2>&1 || exit 1
cd deploy
git rm -r notebooks/*
cd ..
cp -r ./doc/skopt/* deploy

# Move into deployment directory
cd deploy

# Commit changes, allowing empty changes (when unchanged)
echo "Committing and pushing to Github"
git config user.name "Travis-CI"
git config user.email "travis@yoursite.com"
git add -A
git commit --allow-empty -m "Deploying documentation for ${TRAVIS_COMMIT}" || exit 1

# Push to branch
git push origin master > /dev/null 2>&1 || exit 1

echo "Pushed deployment successfully"
exit 0
