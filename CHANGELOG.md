# DeepGNN Changelog

## 0.1 DeepGNN

* Rename and move convert.output to converter.process.converter_process. Dispatchers make argument 'process' default to converter.process.converter_process. Dispatchers move process argument after decoder_type.

* Replace converter and dispatcher's argument "decoder_type" -> "decoder" that accepts Decoder object directly instead of DecoderType enum. Replace DecoderType enum with type hint.

* Make Decoder.decode a generator that yields a node then its outgoing edges in order. The yield format for nodes/edges is (node_id/src, -1/dst, type, weight, features), with features being a list of dense features as ndarrays and sparse features as 2 tuples, coordinates and values.

* Add BinaryWriter as new entry point for NodeWriter, EdgeWriter and alias writers.

* Meta.json files are no longer needed by the converter. Remove meta path argument from MultiWorkerConverter and Dispatchers.

* Support nodes and their outgoing edges on different partitions.
