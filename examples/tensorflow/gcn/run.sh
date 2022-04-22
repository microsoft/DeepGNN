# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -ex

DIR_NAME=$(dirname "$0")

## DEVICE support: ["cpu", "gpu"]
DEVICE=${1:-cpu}

if [[ ${DEVICE} == "gpu" ]]
then
    PLATFORM_DEVICE=--gpu
    env CUDA_VISIBLE_DEVICES=0
fi

CLEANUP=${2:-"no_cleanup"}

DATA_DIR=/tmp/cora/
python -m deepgnn.graph_engine.data.citation --data_dir $DATA_DIR

BASE_DIR=$HOME/tmp/gcn-cora-$(date +"%Y%m%d_%H%M%N")
### =================== EagerTrainer (singler worker) ===================
MODEL_DIR=$BASE_DIR/eager
rm -rf $MODEL_DIR
python $DIR_NAME/main.py \
 --mode train --trainer ps \
 --seed 123 \
 --model_dir $MODEL_DIR \
 --data_dir $DATA_DIR \
 --eager \
 --batch_size 140 \
 --learning_rate 0.01 \
 --epochs 200 \
 --neighbor_edge_types 0 \
 --dropout 0.5 \
 --l2_coef 0.0005 \
 --hidden_dim 16 \
 --feature_idx 0 \
 --feature_dim 1433 \
 --label_idx 1 \
 --label_dim 1 \
 --num_classes 7 \
 --backend snark \
 --converter skip ${PLATFORM_DEVICE}

python $DIR_NAME/main.py \
 --mode evaluate --trainer ps \
 --seed 123 \
 --model_dir $MODEL_DIR \
 --data_dir $DATA_DIR \
 --eager \
 --batch_size 1000 \
 --evaluate_node_files $DATA_DIR/test.nodes \
 --neighbor_edge_types 0 \
 --dropout 0.0 \
 --hidden_dim 16 \
 --feature_idx 0 \
 --feature_dim 1433 \
 --label_idx 1 \
 --label_dim 1 \
 --num_classes 7 \
 --backend snark \
 --converter skip ${PLATFORM_DEVICE}

if [[ "${CLEANUP}" != "no_cleanup" ]]; then
    rm -rf $BASE_DIR
fi
