# DeepGNN note, code in this package was taken from commit 1bee84f6eb75ed7e39e34601bfdd66d79cafe99a.
# https://github.com/microsoft/DeepSpeedExamples/tree/1bee84f6eb75ed7e39e34601bfdd66d79cafe99a/BingBertSquad/turing
# Several trivial changes were made on top:
# 1. Removed the dependency for `deepspeed_config` in nvidia_modeling.py.
# 2. Removed `max_seq_length` from the parameter of DeepSpeedTransformerConfig in nvidia_modeling.py
# 3. Replaced `dense_act` with `dense` in nvidia_modeling.py to make it compatible to checkpoint published by transformers@huggingface.
