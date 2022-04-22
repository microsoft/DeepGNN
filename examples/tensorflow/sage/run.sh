# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -ex

DIR_NAME=$(dirname "$0")

## DEVICE support: ["cpu", "gpu"]
DEVICE=${1:-cpu}
STORAGE_TYPE=${2:-memory}
USE_HADOOP=${3:-no}

if [[ ${DEVICE} == "gpu" ]]
then
    PLATFORM_DEVICE=--gpu
    env CUDA_VISIBLE_DEVICES=0
fi

DATA_DIR=/tmp/ppi
python -m deepgnn.graph_engine.data.ppi --data_dir $DATA_DIR

## ---------------- GraphSAGE supervised -------------------
MODEL_DIR=$HOME/tmp/graphsage_$(date +"%Y%m%d_%H%M%N")
rm -rf $MODEL_DIR
## support agg_type: mean, maxpool, lstm
AGG_TYPE=mean

python $DIR_NAME/main.py --eager --seed 123 \
 --model_dir $MODEL_DIR --data_dir $DATA_DIR \
 --mode train --node_files $DATA_DIR/train.nodes \
 --epochs 10 --batch_size 512 \
 --neighbor_edge_types 0 --num_samples 25,10 \
 --feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121 \
 --layer_dims 128,128 --num_classes 121 \
 --loss_name sigmoid --agg_type $AGG_TYPE --backend snark --converter skip ${PLATFORM_DEVICE} \
 --storage_type $STORAGE_TYPE

python $DIR_NAME/main.py --eager --seed 123 \
 --model_dir $MODEL_DIR --data_dir $DATA_DIR \
 --mode evaluate --node_files $DATA_DIR/val.nodes \
 --neighbor_edge_types 0,1 --num_samples 25,10 \
 --feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121 \
 --layer_dims 128,128 --num_classes 121 \
 --loss_name sigmoid --agg_type $AGG_TYPE \
 --log_save_steps 1 --summary_save_steps 1 --backend snark --converter skip ${PLATFORM_DEVICE}

python $DIR_NAME/main.py --eager --seed 123 \
 --model_dir $MODEL_DIR --data_dir $DATA_DIR \
 --mode evaluate --node_files $DATA_DIR/test.nodes \
 --neighbor_edge_types 0,1 --num_samples 25,10 \
 --feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121 \
 --layer_dims 128,128 --num_classes 121 \
 --loss_name sigmoid --agg_type $AGG_TYPE \
 --log_save_steps 1 --summary_save_steps 1 --backend snark --converter skip ${PLATFORM_DEVICE}

rm -rf $MODEL_DIR

## ---------------- GraphSAGE unsupervised -------------------
MODEL_DIR=$HOME/tmp/graphsage_unsup_$(date +"%Y%m%d_%H%M%N")
rm -rf $MODEL_DIR
## support agg_type: mean, maxpool, lstm
AGG_TYPE=mean

python $DIR_NAME/main_unsup.py --seed 123 \
 --model_dir $MODEL_DIR \
 --data_dir $DATA_DIR \
 --mode train \
 --batch_size 512 \
 --learning_rate 0.001 \
 --epochs 20 \
 --node_types 0 \
 --feature_idx 1 \
 --feature_dim 50 \
 --num_samples 25,10 \
 --layer_dims 128,128 \
 --neighbor_edge_types 0 \
 --negative_node_types 0 \
 --negative_sample_weight 1.0 \
 --negative_num 20 \
 --loss_name xent \
 --agg_type $AGG_TYPE \
 --backend snark \
 --converter skip ${PLATFORM_DEVICE}

rm -rf $MODEL_DIR

# HDFS example
if [[ "$USE_HADOOP" == "yes" ]]; then
    if [[ -d "tools" ]]; then
        . tools/azurepipeline/hdfs_setup.sh
    else
        . ../../hdfs_setup.sh
    fi
    python ${DIR_NAME}/main.py  \
        --eager --seed 123 \
        --model_dir $MODEL_DIR --data_dir file://$DATA_DIR \
        --mode train --node_files $DATA_DIR/train.nodes \
        --epochs 10 --batch_size 512 \
        --neighbor_edge_types 0 --num_samples 25,10 \
        --feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121 \
        --layer_dims 128,128 --num_classes 121 \
        --loss_name sigmoid --agg_type $AGG_TYPE --backend snark --converter skip ${PLATFORM_DEVICE} \
        --stream
fi
