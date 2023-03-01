export SNARK_LIB_PATH="$(pwd)/src/cc/lib/libwrapper.so"
python examples/pytorch/hetgnn/main.py > log.txt
echo "$(cat log.txt)"
if [ "$(grep -ic '0.919048 | 1.32412' log.txt)" != "1" ]; then exit 3; fi
