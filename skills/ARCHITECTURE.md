# DeepGNN Architecture Deep Dive

How the repository is structured and how DeepGNN delivers on its goals: distributed GNN training/inference, custom GNN design, online sampling, automatic graph partitioning, and high performance.

---

## Three-Layer Architecture

```
┌───────────────────────────────────────────────────┐
│  Layer 3: Model & Training                        │
│  PyTorch (DDP, Horovod, FP16) / TensorFlow (PS)  │
│  Example models: GraphSAGE, GCN, GAT, HAN, TGN   │
├───────────────────────────────────────────────────┤
│  Layer 2: Python Graph Engine Wrapper              │
│  Samplers, MultiHop, GraphDataset, query_fn        │
│  Backends: local (in-process) / distributed (GRPC) │
├───────────────────────────────────────────────────┤
│  Layer 1: C++ Native Graph Engine ("snark")        │
│  Partitioned storage, alias-table sampling,        │
│  GRPC server, temporal snapshots, HDFS support     │
└───────────────────────────────────────────────────┘
```

Data flows bottom-up: raw graph data is loaded into the C++ engine, Python wrappers expose sampling and feature APIs, and the training layer consumes mini-batches produced by user-defined query functions.

---

## Layer 1 — C++ Graph Engine (`src/cc/lib/`)

The performance-critical foundation. All graph data lives here, exposed to Python via a C ABI (ctypes).

### Core Data Model (`graph/graph.h`)

The central `Graph` class (namespace `snark`) manages:

```cpp
class Graph {
    std::vector<Partition> m_partitions;           // Sharded graph data
    absl::flat_hash_map<NodeId, uint64_t> m_node_map;  // Global→local ID mapping
    Metadata m_metadata;                           // Replicated metadata
};
```

Key APIs:
- **Node queries**: `GetNodeType()`, `GetNodeFeature()`, `node_types()`
- **Neighbor queries**: `NeighborCount()`, `FullNeighbor()`, `SampleNeighbor()`
- **Feature fetching**: Dense, sparse, and string features with timestamps
- **Advanced**: Random walks, PPR sampling, temporal snapshots

### Partition Storage (`graph/partition.h`)

Each partition holds a contiguous slice of the graph:

```cpp
struct Partition {
    std::vector<Type> m_node_types;           // Node type array
    std::vector<NodeId> m_edge_destination;   // Edge destinations
    std::vector<float> m_edge_weights;        // Edge weights (optional)
    std::vector<uint64_t> m_neighbors_index;  // Per-node neighbor offset

    // Features (indexed)
    std::shared_ptr<BaseStorage<uint8_t>> m_node_features;
    std::vector<uint64_t> m_node_feature_index;
};
```

### Storage Backends (`graph/storage.h`)

| Backend | Description |
|---|---|
| `MemoryStorage` | All data in RAM — fastest |
| `HDFSStorage` | Distributed filesystem for massive graphs |
| `HDFSStreamStorage` | Streaming mode — memory-bounded |
| Disk-based | For memory-constrained environments |

### Samplers (`graph/sampler.h`)

Three sampling techniques, all optimized with alias tables:

1. **Weighted** (`WeightedNodeSampler` / `WeightedEdgeSampler`)
   - p(item) ∝ weight
   - Alias tables: O(1) per sample, O(n log n) preprocessing

2. **Uniform** (`UniformNodeSampler` / `UniformEdgeSampler`)
   - With and without replacement
   - Without replacement uses reservoir sampling (Algorithm L)

3. **Per-partition composition**: `SamplerFactory` → `AbstractSamplerFactory` → `SamplerImpl` → per-partition sampler instances

### Type System (`graph/types.h`)

```cpp
using NodeId = int64_t;              // 64-bit node identifiers
using Type = int32_t;                // Node/edge type labels
using Timestamp = int64_t;           // For temporal graphs
using FeatureMeta = pair<FeatureId, FeatureSize>;
```

Supports typed, temporal, heterogeneous graphs.

### Python Binding (`py_graph.h`, `py_graph.cc`)

Flat C interface for ctypes:

```c
// Local graph creation
CreateLocalGraph(meta_location, partition_count, partitions, storage_type)

// Distributed GRPC client
CreateRemoteClient(servers, ssl_cert, num_threads)

// Sampling
WeightedSampleNeighbor(nodes, edge_types, count, seed)
UniformSampleNeighbor(nodes, edge_types, count, without_replacement)

// Features
GetNodeFeature(node_ids, timestamps, features)
GetEdgeFeature(edges, features)
```

---

## Layer 2 — Python Graph Engine (`src/python/deepgnn/graph_engine/`)

Wraps the C++ engine into clean Python abstractions.

### Abstract API (`_base.py`)

```python
class Graph(abc.ABC):
    def sample_nodes(size, node_types, strategy) → np.ndarray
    def sample_edges(size, edge_types, strategy) → np.ndarray
    def sample_neighbors(nodes, edge_types, count, strategy) → (neighbors, weights, types)
    def node_features(nodes, features, dtype) → np.ndarray
    def edge_features(edges, features, dtype) → np.ndarray
    def random_walk(nodes, metapath, walk_len, p, q) → np.ndarray
    def node_count(types) → int
    def edge_count(types) → int
```

**Sampling strategies** (enum):

| Strategy | Description |
|---|---|
| `Weighted` | Probability proportional to edge/node weight |
| `Random` | Uniform with replacement |
| `RandomWithoutReplacement` | Reservoir sampling |
| `TopK` | Top-K by weight |
| `PPRGo` | Personalized PageRank |
| `LastN` | Temporal: most recent N edges |

### Two Concrete Backends

| | Local (`snark/local.py`) | Distributed (`snark/distributed.py`) |
|---|---|---|
| **How** | Loads partitions in-process via ctypes | Connects to GRPC servers via ctypes |
| **When** | Single machine, data fits in RAM | Multi-machine, massive graphs |
| **API** | Same `Graph` interface | Same `Graph` interface (transparent) |

### Key Modules

**Samplers** (`samplers.py`):
- `GENodeSampler` — iterates through nodes in batches across epochs; auto-discovers node count from the graph
- `GEEdgeSampler` — analogous for edges; returns `[batch_size, 3]` arrays (src, dst, type)
- Both support `data_parallel_index` for sharding iteration across distributed workers

**Multi-hop Sampling** (`multihop.py`):
```python
def sample_fanout(graph, nodes, metapath, fanouts, default_node):
    # Hierarchical neighborhood expansion
    # fanouts=[25, 10] → 25 one-hop neighbors, then 10 two-hop per one-hop node
    # Returns: [nodes, hop0_neighbors, hop1_neighbors, ...], weights, types
```
Powers GraphSAGE-style mini-batch construction.

**Graph Operations** (`graph_ops.py`):
- `sub_graph()` — extract k-hop subgraph with adjacency matrix (used by GCN, GAT)
- `sample_out_edges()` — sample outgoing edges with optional features
- `gen_skipgrams()` — generate context pairs for embedding training

**Dataset** (`graph_dataset.py`):
```python
class DeepGNNDataset:
    def __init__(sampler_class, query_fn, backend, num_workers, batch_size, epochs, ...):
        self.graph = backend.graph
        self.sampler = sampler_class(graph=graph, ...)
        self.query_fn = query_fn

    def __iter__():
        for item_ids in sampler:           # [batch_size] node/edge IDs
            batch = query_fn(graph, item_ids)  # user-defined transform
            yield batch
```

### Backend Configuration (`backends/options.py`)

```python
class BackendOptions:
    backend: BackendType              # SNARK or CUSTOM
    graph_type: GraphType             # LOCAL or REMOTE
    data_dir: str                     # Path to partitioned graph
    storage_type: PartitionStorageType  # memory or disk
    servers: List[str]                # GRPC addresses (distributed)
    enable_ssl: bool                  # TLS for GRPC
```

---

## The Query Function Pattern

The central design pattern in DeepGNN. Users provide a `query_fn` that transforms sampled IDs into model-ready tensors, decoupling data loading from model architecture:

```
Sampler (what)          query_fn (how)              Model (train)
─────────────    →    ─────────────────    →    ─────────────
[node IDs]           graph.node_features()       forward()
[edge IDs]           graph.sample_neighbors()    loss()
                     graph_ops.sub_graph()        backward()
                     custom logic
```

Example for GraphSAGE link prediction:

```python
def query(graph, edge_batch):
    src, dst = edge_batch[:, 0], edge_batch[:, 1]
    neg = np.random.randint(0, num_nodes, ...)       # negative samples
    nodes, cols, rows = multi_hop_sample(src, dst)    # fan-out neighbors
    features = graph.node_features(nodes, feat_meta)  # fetch features
    return (features, cols, rows, edge_label_index, edge_label)
```

Example for GCN node classification:

```python
def query(graph, node_batch):
    nodes, edges, _ = graph_ops.sub_graph(graph, node_batch, edge_types, num_hops=2)
    feat_labels = graph.node_features(nodes, [[feat_id, feat_dim], [label_id, 1]])
    adj = SparseTensor.from_edge_index(edges)
    return (features, gcn_norm(adj), labels)
```

---

## Layer 3 — Training Pipelines

### PyTorch (`src/python/deepgnn/pytorch/`)

```
pytorch/
├── training/
│   ├── trainer.py          # Base single-machine trainer
│   ├── trainer_ddp.py      # PyTorch DistributedDataParallel
│   ├── trainer_hvd.py      # Horovod (multi-framework)
│   ├── trainer_fp16.py     # Mixed-precision training
│   └── factory.py          # Creates trainer by TrainerType enum
├── modeling/
│   └── base_model.py       # Abstract model interface
├── nn/
│   └── gat_conv.py, ...    # Custom GNN layers
└── common/                  # Shared utilities
```

Trainer types:

| Type | Description |
|---|---|
| `BASE` | Single-machine training |
| `DDP` | PyTorch native DistributedDataParallel |
| `HVD` | Horovod (works across frameworks) |
| FP16 | Mixed-precision via PyTorch AMP |

### TensorFlow (`src/python/deepgnn/tf/`)

Parallel structure with TF-specific layers (`tf/layers/`), encoders (`tf/encoders/`), and parameter-server training. Example models: GAT, GCN, HAN, SAGE (supervised, unsupervised, link prediction).

---

## Example Models (`examples/`)

| Model | File | Task | Key technique |
|---|---|---|---|
| GraphSAGE | `examples/pytorch/sage.py` | Link prediction | Multi-hop fan-out sampling, negative sampling |
| GCN | `examples/pytorch/gcn.py` | Node classification | k-hop subgraph extraction, sparse adjacency |
| GAT | `examples/pytorch/gat.py` | Node classification | Attention-compatible adjacency, Ray distribution |
| HetGNN | `examples/pytorch/hetgnn/` | Heterogeneous | Multi-type sampling, type-specific encoders |
| TGN | `examples/pytorch/tgn.py` | Temporal | Temporal neighborhood, time-encoded features |
| TF SAGE | `examples/tensorflow/sage/` | Link prediction + unsupervised | Parameter server, skip-gram loss |
| TF HAN | `examples/tensorflow/han/` | Node classification | Hierarchical attention on heterogeneous graphs |

---

## End-to-End Data Flow

What happens when running `python sage.py`:

```
1. INIT
   ├── Parse args (BackendOptions, model params)
   ├── SnarkLocalBackend → loads partitions into C++ MemoryGraph
   └── Build DeepGNNDataset with GEEdgeSampler + query_fn

2. EACH TRAINING STEP
   ├── GEEdgeSampler
   │     → C++ WeightedSampleEdge → returns [batch_size, 3] edges
   ├── query_fn()
   │   ├── Negative sampling (random node IDs)
   │   ├── Multi-hop neighbor sampling → C++ WeightedSampleNeighbor per hop
   │   ├── Feature fetch → C++ GetNodeFeature
   │   └── Returns (features, edge_index, labels) as tensors
   ├── Model.forward() → SAGEConv message-passing layers
   ├── Loss (binary cross-entropy for link prediction)
   └── Backward + optimizer step

3. EVALUATION
   └── Same pipeline with eval sampler, report metrics
```

---

## Distributed Architecture

### Local Mode

```
Python Process
  ├── DeepGNNDataset + Trainer
  └── local.Client → MemoryGraph (C++, in-process)
       └── Partitions [0..n] in RAM
```

### Distributed Mode

```
Worker 0 (DDPTrainer)           Worker 1 (DDPTrainer)
  └── SynchronizedClient          └── SynchronizedClient
       │ GRPC                          │ GRPC
       ↓                              ↓
  GE Server 0          GE Server 1          GE Server 2
  Partitions [0,3,6]   Partitions [1,4,7]   Partitions [2,5,8]
```

Workers synchronize startup via filesystem markers in a shared `sync_dir`. The `Graph` API is identical in both modes — switching from local to distributed requires only a config change.

---

## How the README Promises Are Delivered

| README Feature | Implementation |
|---|---|
| **Distributed GNN training (CPU+GPU)** | DDP/Horovod trainers + GRPC distributed graph engine with partition-level parallelism |
| **Custom GNN design** | Abstract `BaseModel` + user-defined `query_fn` + composable layers in `pytorch/nn/` and `tf/layers/` |
| **Online Sampling** | Graph Engine loads all data; workers call `sample_neighbors()`/`sample_nodes()` each step via C++ alias-table samplers |
| **Automatic graph partitioning** | C++ Partition abstraction + multi-partition loading + per-partition sampler composition |
| **High performance & scalability** | C++ core with O(1) alias-table sampling, memory-mapped storage, GRPC streaming, prefetch pipeline, HDFS for huge graphs |

---

## Key Design Decisions

1. **C++ graph engine with alias-table sampling** — O(1) per sample makes online sampling practical at scale
2. **Query function abstraction** — users define arbitrary data transforms without touching the training loop or sampling infrastructure
3. **Transparent local/distributed backends** — same code on a laptop or a cluster; only config changes
4. **Per-partition sampler composition** — each partition maintains its own alias table; the framework composes them into a single sampler that respects global weight distributions
5. **Temporal-first type system** — `Timestamp` is a first-class concept in the C++ layer, enabling temporal graph models (TGN) without workarounds
