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

MODEL_DIR=$HOME/tmp/han_local-$(date +"%Y%m%d_%H%M%N")
DATA_DIR=/tmp/cora/
python -m deepgnn.graph_engine.data.citation --data_dir $DATA_DIR

## training, save model checkpoint to $MODEL_DIR
rm -rf $MODEL_DIR
python3 $DIR_NAME/main.py --mode train \
 --model_dir $MODEL_DIR \
 --data_dir $DATA_DIR \
 --training_node_types 0 \
 --edge_types "0;1" \
 --num_nodes 300 \
 --feature_idx 0 \
 --feature_dim 1870 \
 --batch_size 24 \
 --learning_rate 0.001 \
 --epochs 10 \
 --head_num 8 \
 --hidden_dim 128 \
 --label_idx 1 \
 --label_dim 3 \
 --fanouts 10 \
 --seed 123 \
 --backend snark \
 --converter skip ${PLATFORM_DEVICE}


## Evaluation
python3 $DIR_NAME/main.py --mode evaluate \
 --model_dir $MODEL_DIR \
 --data_dir $DATA_DIR \
 --evaluate_node_types 1 \
 --edge_types "0;1" \
 --num_nodes 2000 \
 --feature_idx 0 \
 --feature_dim 1870 \
 --batch_size 24 \
 --head_num 8 \
 --hidden_dim 128 \
 --label_idx 1 \
 --label_dim 3 \
 --fanouts 10 \
 --seed 123 \
 --backend snark \
 --converter skip ${PLATFORM_DEVICE}

## Inference
python3 $DIR_NAME/main.py --mode inference \
 --model_dir $MODEL_DIR \
 --data_dir $DATA_DIR \
 --edge_types "0;1" \
 --num_nodes 2000 \
 --feature_idx 0 \
 --feature_dim 1870 \
 --batch_size 24 \
 --head_num 8 \
 --hidden_dim 128 \
 --fanouts 10 \
 --seed 123 \
 --backend snark \
 --converter skip ${PLATFORM_DEVICE}

if [[ "${CLEANUP}" != "no_cleanup" ]]; then
    rm -rf $MODEL_DIR
fi
