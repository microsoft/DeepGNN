# DeepGNN Changelog

## 0.1 DeepGNN

* Rename and move convert.output to converter.process.converter_process. Dispatchers make argument 'process' default to converter.process.converter_process. Dispatchers move process argument after decoder_type.

* Replace converter and dispatcher's argument "decoder_type" -> "decoder" that accepts Decoder object directly instead of DecoderType enum. Replace DecoderType enum with type hint.
