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

python3 -m deepgnn.graph_engine.data.cora --data_dir $GRAPH

MODEL_DIR=/tmp/model_fix
rm -rf $MODEL_DIR

if [[ "${DEVICE}" == "gpu" ]]
then
    PLATFORM_DEVICE=--gpu
fi

### ===== training =======
python3 ${DIR_NAME}/main.py  \
--data_dir $GRAPH --mode train --seed 123 \
--backend snark --graph_type local --converter skip \
--batch_size 64 --learning_rate 0.005 --num_epochs 100 \
--node_type 0 --max_id -1 \
--model_dir $MODEL_DIR --metric_dir $MODEL_DIR --save_path $MODEL_DIR \
--feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121 --algo $ALGO \
--log_by_steps 1 --use_per_step_metrics $PLATFORM_DEVICE --storage_type $STORAGE_TYPE

python3 ${DIR_NAME}/main.py  \
--data_dir $GRAPH --mode evaluate \
--backend snark --graph_type local --converter skip \
--batch_size 1000 \
--sample_file /tmp/cora/test.nodes --node_type 0 --max_id -1 \
--feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121 --algo $ALGO \
--model_dir $MODEL_DIR --metric_dir $MODEL_DIR --save_path $MODEL_DIR \
--log_by_steps 1 --use_per_step_metrics $PLATFORM_DEVICE

if [[ "$ADL_UPLOADER" == "no" ]]; then
    python3 ${DIR_NAME}/main.py  \
    --data_dir $GRAPH --mode inference \
    --backend snark --graph_type local --converter skip \
    --batch_size 1000 \
    --sample_file /tmp/cora/test.nodes --node_type 0 \
    --feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121 --algo $ALGO \
    --model_dir $MODEL_DIR --metric_dir $MODEL_DIR --save_path $MODEL_DIR \
    --log_by_steps 1 --use_per_step_metrics $PLATFORM_DEVICE
else
    python3 ${DIR_NAME}/main.py  \
    --data_dir $GRAPH --mode inference \
    --backend snark --graph_type local --converter skip \
    --batch_size 1000 \
    --sample_file /tmp/cora/test.nodes --node_type 0 \
    --feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121 --algo $ALGO \
    --model_dir $MODEL_DIR --metric_dir $MODEL_DIR --save_path /integration_test/test_adl_uploader \
    --log_by_steps 1 --use_per_step_metrics --enable_adl_uploader --uploader_store_name snrgnndls --uploader_process_num 2 --uploader_threads_num 10 $PLATFORM_DEVICE
fi

rm -rf $MODEL_DIR

# HDFS example
if [[ "$USE_HADOOP" == "yes" ]]; then
    if [[ -d "tools" ]]; then
        . tools/azurepipeline/hdfs_setup.sh
    else
        . ../../hdfs_setup.sh
    fi
    python3 ${DIR_NAME}/main.py  \
    --data_dir file://$GRAPH --mode train --seed 123 \
    --backend snark --graph_type local --converter skip \
    --batch_size 140 --learning_rate 0.005 --num_epochs 100 \
    --node_type 0 --max_id -1 \
    --model_dir $MODEL_DIR --metric_dir $MODEL_DIR --save_path $MODEL_DIR \
    --feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121 --algo $ALGO \
    --log_by_steps 1 --use_per_step_metrics $PLATFORM_DEVICE \
    --stream
fi

rm -rf $MODEL_DIR