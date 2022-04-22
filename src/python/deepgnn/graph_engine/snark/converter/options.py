# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Snark converter option."""
from enum import Enum


class DataConverterType(Enum):
    """DeepGNN's graph engine data converter type.

    Supported:
        Skip:  binary data is already converted and no need to convert any more.
    """

    SKIP = "skip"
    LOCAL = "local"

    def __str__(self):
        """Convert to string."""
        return self.value


class ConverterOptions:
    """All the data converter related configurations are here.

    Converters supported:
        Skip
    """

    def __init__(self, params):
        """Init the coverter option."""
        # default values.
        self.converter = DataConverterType.SKIP

        for arg in vars(params):
            if hasattr(self, arg):
                setattr(self, arg, getattr(params, arg))
