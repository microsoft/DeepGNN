# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -ex

DIR_NAME=$(dirname "$0")

pip install --upgrade pip
pip install -r $DIR_NAME/requirements.txt

cat <<EOT > $DIR_NAME/inventory/hosts
all:
  children:
    linux:
      hosts:
        localhost:
          ansible_host: 127.0.0.1
      vars:
        ansible_connection: local
EOT

cat $DIR_NAME/inventory/hosts
echo "=================="
cat $DIR_NAME/playbooks/deploy.yml

cd $DIR_NAME
python ./distribute.py --deploy_infra
python -m deepgnn.graph_engine.data.citation --data_dir /tmp/cora
python ./distribute.py --port 12345 --data_dir /tmp/cora --host_count 1 --partition_count 1

sleep 10

rm -fr /tmp/model_fix
cd $DIR_NAME/../../pytorch/gat

python main.py  \
--mode train --seed 123 \
--backend snark --graph_type remote --converter skip --servers "localhost:12345" --skip_ge_start True \
--batch_size 140 --learning_rate 0.005 --num_epochs 180 \
--sample_file /tmp/cora/train.nodes --node_type 0 \
--model_dir /tmp/model_fix --metric_dir /tmp/model_fix --save_path /tmp/model_fix \
--eval_file /tmp/cora/test.nodes --eval_during_train_by_steps 1 \
--feature_idx 0 --feature_dim 1433 --label_idx 1 --label_dim 1 \
--head_num 8,1 --num_classes 7 --neighbor_edge_types 0 --attn_drop 0.6 --ffd_drop 0.6 \
--log_by_steps 1 --use_per_step_metrics
