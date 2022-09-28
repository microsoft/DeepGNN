# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -ex

## DEVICE support: ["cpu", "gpu"]
DEVICE=${1:-cpu}

DIR_NAME=$(dirname "$0")

GRAPH=/tmp/cora
#rm -fr $GRAPH

#python -m deepgnn.graph_engine.data.citation --data_dir $GRAPH

if [[ ${DEVICE} == "gpu" ]]
then
    PLATFORM_DEVICE=--gpu
fi

MODEL_DIR=/tmp/model_fix
rm -rf $MODEL_DIR

### ===== training =======
python ${DIR_NAME}/main.py --data_dir $GRAPH --mode train --seed 123 --converter skip \
--batch_size 140 --learning_rate 0.005 --num_epochs 10 --max_id -1 --node_type_count 1 \
--model_dir $MODEL_DIR --metric_dir $MODEL_DIR --save_path $MODEL_DIR --max_id 140 \
--feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121 \
--log_by_steps 1 --use_per_step_metrics # $PLATFORM_DEVICE

#python ${DIR_NAME}/main.py  \
#--data_dir $GRAPH --mode evaluate --seed 123 --converter skip \
#--batch_size 140 --learning_rate 0.005 --max_id -1 --node_type_count 3 \
#--model_dir $MODEL_DIR --metric_dir $MODEL_DIR --save_path $MODEL_DIR \
#--feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121 --max_id 140 \
#--log_by_steps 1 --use_per_step_metrics $PLATFORM_DEVICE

#python ${DIR_NAME}/main.py  \
#--data_dir $GRAPH --mode inference --seed 123 --converter skip \
#--batch_size 140 --learning_rate 0.005 \
#--sample_file /tmp/cora/test.nodes.hetgnn --node_type 0 --node_type_count 3 \
#--model_dir $MODEL_DIR --metric_dir $MODEL_DIR --save_path $MODEL_DIR \
#--feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121 \
#--log_by_steps 1 --use_per_step_metrics $PLATFORM_DEVICE
