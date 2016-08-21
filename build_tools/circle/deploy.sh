#!/bin/bash
# Copying to github pages
echo "Copying built files"
git clone -b master "https://${GH_TOKEN}@github.com/scikit-optimize/scikit-optimize.github.io.git" deploy > /dev/null 2>&1 || exit 1
cd deploy
git rm -r notebooks/*
cd ..
cp -r ${HOME}/doc/skopt/* deploy

# Move into deployment directory
cd deploy

# Commit changes, allowing empty changes (when unchanged)
echo "Committing and pushing to Github"
git config user.name "Travis-CI"
git config user.email "travis@yoursite.com"
git add -A
git commit --allow-empty -m "Deploying documentation for ${TRAVIS_COMMIT}" || exit 1

# Push to branch
git push origin > /dev/null 2>&1 || exit 1

echo "Pushed deployment successfully"
exit 0
