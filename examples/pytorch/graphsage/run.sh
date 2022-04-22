# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -ex

ALGO=${1:-supervised}
ADL_UPLOADER=${2:-no}
## DEVICE support: ["cpu", "gpu"]
DEVICE=${3:-cpu}
STORAGE_TYPE=${4:-memory}
USE_HADOOP=${5:-no}

DIR_NAME=$(dirname "$0")

GRAPH=/tmp/cora
rm -fr $GRAPH

python -m deepgnn.graph_engine.data.citation --data_dir $GRAPH

MODEL_DIR=/tmp/model_fix
rm -rf $MODEL_DIR

if [[ "${DEVICE}" == "gpu" ]]
then
    PLATFORM_DEVICE="--gpu=true"
fi

### ===== training =======
python ${DIR_NAME}/main.py --conf-dir conf \
    --data_dir=$GRAPH --algo=$ALGO --model_dir=$MODEL_DIR \
    --metric_dir=$MODEL_DIR --save_path=$MODEL_DIR $PLATFORM_DEVICE \
    --storage_type=$STORAGE_TYPE

python ${DIR_NAME}/main.py --conf-dir conf \
    --mode=evaluate --batch_size=1000 --sample_file=/tmp/cora/test.nodes \
    --data_dir=$GRAPH --algo=supervised --model_dir=$MODEL_DIR \
    --metric_dir=$MODEL_DIR --save_path=$MODEL_DIR $PLATFORM_DEVICE

if [[ "$ADL_UPLOADER" == "no" ]]; then
    python ${DIR_NAME}/main.py --conf-dir conf \
        --mode=inference --batch_size=1000 --sample_file=/tmp/cora/test.nodes \
        --data_dir=$GRAPH --algo=$ALGO --model_dir=$MODEL_DIR \
        --metric_dir=$MODEL_DIR --save_path=$MODEL_DIR $PLATFORM_DEVICE
else
    python ${DIR_NAME}/main.py --conf-dir conf +deepgnn=adl_uploader \
        --mode=inference --batch_size=1000 --sample_file=/tmp/cora/test.nodes \
        --data_dir=$GRAPH --algo=$ALGO --model_dir=$MODEL_DIR \
        --metric_dir=$MODEL_DIR $PLATFORM_DEVICE
fi

rm -rf $MODEL_DIR

# HDFS example
if [[ "$USE_HADOOP" == "yes" ]]; then
    if [[ -d "tools" ]]; then
        . tools/azurepipeline/hdfs_setup.sh
    else
        . ../../hdfs_setup.sh
    fi
    python ${DIR_NAME}/main.py --conf-dir conf \
        --data_dir=file://$GRAPH --algo=$ALGO --model_dir=$MODEL_DIR \
        --metric_dir=$MODEL_DIR --save_path=$MODEL_DIR $PLATFORM_DEVICE \
        --stream=true
fi

rm -rf $MODEL_DIR
