# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -ex

DIR_NAME=$(dirname "$0")

DATA_DIR=/tmp/ppi
python -m deepgnn.graph_engine.data.ppi --data_dir $DATA_DIR

AGG_TYPE=mean

## ---------------- GraphSAGE supervised (eager + distributed horovod )-------------------
## training: horovod(--eager), CPU, 2 workers
MODEL_DIR=$HOME/tmp/graphsage_hvd_w2_$(date +"%Y%m%d_%H%M%N")
rm -rf $MODEL_DIR
python $DIR_NAME/../../../deploy/local.py \
 --TRAINER TF_HVD \
 --OUTPUT_DIR $MODEL_DIR \
 --CHECK_INTERVAL 5 \
 --NUM_WORKERS_PER_MACHINE 2 \
 --MAIN main.py --eager --seed 123 \
 --model_dir $MODEL_DIR --data_dir $DATA_DIR \
 --mode train --node_types 0 \
 --epochs 1 --batch_size 512 \
 --neighbor_edge_types 0 --num_samples 25,10 \
 --feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121 \
 --layer_dims 128,128 --num_classes 121 \
 --backend snark --converter skip \
 --loss_name sigmoid --agg_type $AGG_TYPE

rm -rf $MODEL_DIR


## ---------------- GraphSAGE unsupervsied (non-earger + distributed horovod )-------------------
## training: TF_HVD, CPU, 2 workers
MODEL_DIR=$HOME/tmp/graphsage_hvd_w2_$(date +"%Y%m%d_%H%M%N")
rm -rf $MODEL_DIR
python $DIR_NAME/../../../deploy/local.py \
 --TRAINER TF_HVD \
 --OUTPUT_DIR $MODEL_DIR \
 --CHECK_INTERVAL 5 \
 --NUM_WORKERS_PER_MACHINE 2 \
 --MAIN main_unsup.py --seed 123 \
 --model_dir $MODEL_DIR \
 --data_dir $DATA_DIR \
 --mode train \
 --batch_size 512 \
 --learning_rate 0.001 \
 --epochs 1 \
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
 --backend snark --converter skip \
 --agg_type $AGG_TYPE

rm -rf $MODEL_DIR

## ---------------- GraphSAGE unsupervsied (non-earger + distributed PS )-------------------
## training: TF_PS, CPU, 2 workers
MODEL_DIR=$HOME/tmp/graphsage_ps_w2_$(date +"%Y%m%d_%H%M%N")
rm -rf $MODEL_DIR
python $DIR_NAME/../../../deploy/local.py \
 --TRAINER TF_PS \
 --OUTPUT_DIR $MODEL_DIR \
 --CHECK_INTERVAL 5 \
 --NUM_WORKERS_PER_MACHINE 2 \
 --MAIN main_unsup.py --seed 123 \
 --model_dir $MODEL_DIR \
 --data_dir $DATA_DIR \
 --mode train \
 --batch_size 512 \
 --learning_rate 0.001 \
 --epochs 1 \
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
 --backend snark --converter skip \
 --agg_type $AGG_TYPE

rm -rf $MODEL_DIR
