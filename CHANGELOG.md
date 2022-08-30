# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Breaking. Added version checks for binary data. Requires to convert graph data or add v1 at the top of meta files.

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
