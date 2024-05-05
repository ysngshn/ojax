#!/usr/bin/env bash

# build Sphinx docs to html
make clean html &&

# copy generated html to docs/
cp -rvf _build/html/* ../docs &&
cp -vf _build/html/.nojekyll ../docs &&

# remove :orphan: which mess up readme rendering
sed -i '4d' ../readme.rst

