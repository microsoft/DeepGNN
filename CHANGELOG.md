# DeepGNN Changelog

## 0.1 DeepGNN

* Dispatchers make argument 'process' default to converter.process.convert_process and set the argument after decoder type.

* Replace converter and dispatcher's argument "decoder_type" -> "decoder" that accepts Decoder object directly instead of DecoderType enum. Replace DecoderType enum with type hint.

* Make Decoder.decode a generator that yields a node then its outgoing edges in order. Yield format is -1/src node_id/dst type weight features, with features being a list of dense features as ndarrays and sparse features as 2 tuples, coordinates and values.
