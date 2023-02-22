export SNARK_LIB_PATH="$(pwd)/src/cc/lib/libwrapper.so"
python examples/pytorch/graphsage/main.py > log.txt
echo "$(cat log.txt)"
if [ "$(grep -ic 'Epoch 190 F1Score: 0.1786 Loss: 4.6198' log.txt)" != "1" ]; then exit 3; fi
