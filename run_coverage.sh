#!/bin/sh -
python -m coverage run --source ojax -m unittest discover -s tests/ -v
python -m coverage report
python -m coverage html
