#!/usr/bin/env bash
cd sphinx &&
bash doc-build.sh &&
cd .. &&
bash run_coverage.sh &&
cd .. &&
echo "==== All done! ===="
