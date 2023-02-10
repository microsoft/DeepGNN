# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -ex

DIR_NAME=$(dirname "$0")

DEVICE=${1:-cpu}
if [[ ${DEVICE} == "gpu" ]]
then
    PLATFORM_DEVICE=--gpu
fi

GRAPH=/tmp/cora
python -m deepgnn.graph_engine.data.citation --data_dir $GRAPH

MODEL_DIR=/tmp/model_fix
rm -rf $MODEL_DIR

### ===== training =======
python ${DIR_NAME}/main.py  \
--data_dir $GRAPH --mode train --seed 123 \
--backend snark --graph_type local --converter skip \
--batch_size 140  --num_epochs 180 \
--sample_file /tmp/cora/train.nodes --node_type 0 \
--model_dir $MODEL_DIR --metric_dir $MODEL_DIR --save_path $MODEL_DIR \
--feature_idx 0 --feature_dim 1433 --label_idx 1 --label_dim 1 --num_classes 7 \
${PLATFORM_DEVICE}

python ${DIR_NAME}/main.py  \
--backend snark --graph_type local --converter skip \
--data_dir $GRAPH --mode evaluate \
--batch_size 1000 \
--sample_file /tmp/cora/test.nodes --node_type 0 \
--feature_idx 0 --feature_dim 1433 --label_idx 1 --label_dim 1 --num_classes 7 \
--model_dir $MODEL_DIR --metric_dir $MODEL_DIR --save_path $MODEL_DIR \
${PLATFORM_DEVICE}
