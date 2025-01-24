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
python -m deepgnn.graph_engine.data.cora --data_dir $DATA_DIR

BASE_DIR=$HOME/tmp/gat-cora-$(date +"%Y%m%d_%H%M%N")
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
 --learning_rate 0.005 \
 --epochs 300 \
 --neighbor_edge_types 0 \
 --attn_drop 0.6 \
 --ffd_drop 0.6 \
 --head_num 8,1 \
 --l2_coef 0.0005 \
 --hidden_dim 8 \
 --feature_idx 0 \
 --feature_dim 1433 \
 --label_idx 1 \
 --label_dim 1 \
 --num_classes 7 \
 --backend snark \
 --converter skip ${PLATFORM_DEVICE}

### =================== HorovodEagerTrainer (singler worker) ===================
MODEL_DIR=$BASE_DIR/hvd
rm -rf $MODEL_DIR
python $DIR_NAME/main.py \
 --mode train --trainer hvd \
 --seed 123 \
 --model_dir $MODEL_DIR \
 --data_dir $DATA_DIR \
 --eager \
 --batch_size 140 \
 --learning_rate 0.005 \
 --epochs 300 \
 --neighbor_edge_types 0 \
 --attn_drop 0.6 \
 --ffd_drop 0.6 \
 --head_num 8,1 \
 --l2_coef 0.0005 \
 --hidden_dim 8 \
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
