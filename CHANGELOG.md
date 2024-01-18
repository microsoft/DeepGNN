# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.62] - 2024-01-18

### Fixed
- Fixes edge feature fetching when edges are not sorted by destination id.

## [0.1.61] - 2022-10-12

### Added
- Add `return_edge_created_ts` argument to neighbor sampling methods to return timestamps when edges connecting nodes were created.
- `MOOC` temporal dataset.
- TGN example.
- GCN example.

- Add PyG remote backend example.

### Fixed
- Uniform sampling works in temporal graphs.
- ADL path parsing to download graph data.

### Changed
- Changed pytorch examples to be self contained and use [Ray](https://www.ray.io/) for distributed training.

### Removed
- link prediction and knowledgegraph examples
- deepgnn-torch/tf are no longer published

## [0.1.60] - 2022-04-18

### Added
- Breaking. Temporal graph support. Custom decoders must add 2 optional integers in returned tuple in  `decode` method, representing `created_at` and `removed_at` fields. Metadata file must have a `watermark` field.

- Last N created neighbors sampling method for temporal graphs.

### Changed
- Change generated file meta.txt to meta.json in json format.

## [0.1.59] - 2022-03-29

### Added
- All `DistributedGraph` config options (e.g. `grpc_options`, `num_threads`, ...) are exposed to `DistributedClient` and `BackendOptions`

## [0.1.58] - 2022-02-15

### Added
- Add usage example for Ray Train, see [docs/torch/ray_usage.rst](https://github.com/microsoft/DeepGNN/tree/main/docs/torch/ray_usage.rst).

- Add documentation for Ray Data usage, see [tutorial](https://github.com/microsoft/DeepGNN/tree/main/docs/graph_engine/dataset.rst) and [example](https://github.com/microsoft/DeepGNN/tree/main/docs/graph_engine/ray_usage_advanced.rst)

- Add Reddit dataset download tool at deepgnn.graph_engine.data.reddit.

- Added `grpc_options` to distributed client to control service config.

- Added `ppr-go` neighbor sampling strategy.

### Fixed
- Implement del method to release C++ client and server. Important for ray actors, because they create numerous clients during training.

- If sparse feature values present on multiple servers, then only one will be returned with source picked randomly.

### Removed
- Remove ALL_NODE_TYPE, ALL_EDGE_TYPE, __len__ and __iter__ from Graph API.

## [0.1.57] - 2022-12-15

### Changed
- Breaking. Rename get_feature_type -> get_python_type.

## [0.1.56] - 2022-11-02

### Added
- Add new converter input format "EdgeList" with EdgeListDecoder. Format has nodes and edges on separate lines, is smaller and faster to convert.

- Breaking. Added version checks for binary data. Requires to convert graph data or add v1 at the top of meta files.

- Add migrate script to pull to new version of deepgnn.

- Add debug mode to MultiWorkersConverter, using debug=True will now disable multiprocessing and show error messages.

- Load graph partitions from separate folders.

### Changed
- Breaking. Remove FeatureType enum, replace with np.dtype. FeatureType.BINARY -> np.uint8, FeatureType.FLOAT -> np.float32, FeatureType.INT64 -> np.int64.

- Rename function deepgnn.graph_engine.data.to_json_node -> deepgnn.graph_engine.data.to_edge_list_node and update functionality accordingly.

## [0.1.55] - 2022-08-26

### Added
- Support nodes and their outgoing edges on different partitions.

- Adds neighbor count method to graph.

### Fixed
- Return empty indices and values for missing sparse features.

## [0.1.54] - 2022-08-04

### Fixed
- Don't record empty sparse features and log warning if sparse features were requested, but dense features are stored.

## [0.1.53] - 2022-08-02

### Fixed
- JSON/TSV converter didn't sort edges by types resulted in incorrect sampling.

## [0.1.52] - 2022-07-27

### Changed
- Rename and move convert.output to converter.process.converter_process. Dispatchers make argument 'process' default to converter.process.converter_process. Dispatchers move process argument after decoder_type.

- Replace converter and dispatcher's argument "decoder_type" -> "decoder" that accepts Decoder object directly instead of DecoderType enum. Replace DecoderType enum with type hint.

- Make Decoder.decode a generator that yields a node then its outgoing edges in order. The yield format for nodes/edges is (node_id/src, -1/dst, type, weight, features), with features being a list of dense features as ndarrays and sparse features as 2 tuples, coordinates and values.

### Added
- Add BinaryWriter as new entry point for NodeWriter, EdgeWriter and alias writers.

### Removed
- Meta.json files are no longer needed by the converter. Remove meta path argument from MultiWorkerConverter and Dispatchers.

### Fixed
- Fill dimensions with 0 for missing features.
