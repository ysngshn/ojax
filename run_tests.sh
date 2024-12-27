#!/bin/sh -
XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python -m unittest discover -s tests/ -v
