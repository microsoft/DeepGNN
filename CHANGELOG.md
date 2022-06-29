# DeepGNN Changelog

## 0.1 DeepGNN

* Dispatchers make argument 'process' default to converter.process.convert_process and set the argument after decoder type.

* Replace converter and dispatcher's argument "decoder_type" -> "decoder_class" that accepts Decoder object directly instead of DecoderType enum. Replace DecoderType enum with type hint.
