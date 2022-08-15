<<<<<<< HEAD
## Graph Convolutional Networks (GCN)
- Reference : https://arxiv.org/abs/1609.02907
- Author's code: https://github.com/tkipf/gcn

### How to run
see [run.sh](./run.sh)


#### Results

| Dataset  | Test Accuracy | Baseline (Paper) |
| -------- | ------------- | ---------------- |
| Cora     | 80.8          | 81.5             |
| Citeseer | 70.4          | 70.3             |


Training time: (200 epochs)
| Dataset  | CPU  | GPU |
| -------- | ---- | --- |
| Cora     | 2.9  | 2.3 |
| Citeseer | 4.0  | 2.5 |

* CPU: E5-2690 v4 @ 2.60GHz (6 cores)
* GPU: P100 16GB

### Run GCN with your graph
* [Prepare Graph Data](../../../docs/advanced/data_spec.md)
=======
## Graph Convolutional Networks (GCN)
- Reference : https://arxiv.org/abs/1609.02907
- Author's code: https://github.com/tkipf/gcn

### How to run
see [run.sh](./run.sh)


#### Results

| Dataset  | Test Accuracy | Baseline (Paper) |
| -------- | ------------- | ---------------- |
| Cora     | 80.8          | 81.5             |
| Citeseer | 70.4          | 70.3             |


Training time: (200 epochs)
| Dataset  | CPU  | GPU |
| -------- | ---- | --- |
| Cora     | 2.9  | 2.3 |
| Citeseer | 4.0  | 2.5 |

* CPU: E5-2690 v4 @ 2.60GHz (6 cores)
* GPU: P100 16GB

### Run GCN with your graph
* [Prepare Graph Data](../../../docs/graph_engine/data_spec.rst)
>>>>>>> b30762e9d10d2ae19e206c7622f6d10554dc84f0
