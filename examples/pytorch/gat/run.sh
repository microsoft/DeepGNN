# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -ex

## DEVICE support: ["cpu", "gpu"]
DEVICE=${1:-cpu}

DIR_NAME=$(dirname "$0")

GRAPH=/tmp/cora
python -m deepgnn.graph_engine.data.citation --data_dir $GRAPH

MODEL_DIR=/tmp/model_fix
rm -rf $MODEL_DIR

if [[ ${DEVICE} == "gpu" ]]
then
    PLATFORM_DEVICE=--gpu
fi

### ===== training =======
python ${DIR_NAME}/main.py  \
--data_dir $GRAPH --mode train --seed 123 \
--converter skip \
--batch_size 140 --learning_rate 0.005 --num_epochs 180 \
--sample_file /tmp/cora/train.nodes --node_type 0 \
--model_dir $MODEL_DIR --metric_dir $MODEL_DIR --save_path $MODEL_DIR \
--eval_file /tmp/cora/test.nodes --eval_during_train_by_steps 1 \
--feature_idx 0 --feature_dim 1433 --label_idx 1 --label_dim 1 \
--head_num 8,1 --num_classes 7 --neighbor_edge_types 0 --attn_drop 0.6 --ffd_drop 0.6 \
--log_by_steps 1 --use_per_step_metrics \
$PLATFORM_DEVICE

python ${DIR_NAME}/main.py  \
--data_dir $GRAPH --mode evaluate \
--converter skip \
--batch_size 1000 \
--sample_file /tmp/cora/test.nodes --node_type 0 \
--feature_idx 0 --feature_dim 1433 --label_idx 1 --label_dim 1 \
--head_num 8,1 --num_classes 7 \
--model_dir $MODEL_DIR --metric_dir $MODEL_DIR --save_path $MODEL_DIR \
--log_by_steps 1 --use_per_step_metrics \
$PLATFORM_DEVICE
