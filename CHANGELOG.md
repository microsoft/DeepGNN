# DeepGNN Changelog

## 0.1 DeepGNN

* Remove FeatureType enum, replace with np.dtype. FeatureType.BINARY -> np.bool8, FeatureType.FLOAT -> np.float32, FeatureType.INT64 -> np.int64. Fetch features parameter feature_type -> dtype.
