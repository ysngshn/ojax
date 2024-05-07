cd sphinx &&
bash doc-build.sh &&
cd ../tests &&
python test_ojax.py &&
cd .. &&
echo "==== All done! ===="
