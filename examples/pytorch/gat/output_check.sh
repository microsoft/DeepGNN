export SNARK_LIB_PATH="$(pwd)/src/cc/lib/libwrapper.so"
python examples/pytorch/gat/main.py > log.txt
echo "$(cat log.txt)"
if [ "$(grep -ic '0.826 | 0.721931' log.txt)" != "1" ]; then exit 3; fi
