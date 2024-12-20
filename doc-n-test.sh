#!/bin/sh -
echo
echo "== building sphinx doc ..."
echo
cd sphinx &&
sh doc-build.sh &&
cd .. &&
echo
echo "== running unittest coverage ..."
echo
sh run_coverage.sh &&
echo
echo "== running mypy ..."
echo
sh run_mypy.sh &&
echo
echo "== running black ..."
echo
sh run_black.sh &&
echo
echo "==== All done! ===="
