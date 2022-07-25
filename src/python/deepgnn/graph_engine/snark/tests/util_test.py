# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit test utils."""
from deepgnn.graph_engine.snark.decoders import JsonDecoder


def linear_encode(
    node_id: int,
    node_type: int,
    node_weight: float,
    node_features: list,
    edges: list,
    buffer: object = None,  # type: ignore
):
    """
    Convert data to a line of the linear format.

    Parameters
    ----------
    node_id: int
    node_type int
    node_weight: float
    node_features: [ndarray, ...]
    edges: [(edge_src: int, edge_dst: int, edge_type: int, edge_weight: float, edge_features[ndarray, ...]), ...]
    """

    def _feature_str(features):
        def get_f(f):
            if f is None:
                return "uint8 0"
            elif isinstance(f, str):
                return f"binary_feature 1 {f}"
            elif isinstance(f, tuple):
                coords, values = f
                if len(coords.shape) == 1:
                    coordinates_str = " ".join(map(str, coords))
                    length = f"{coords.shape[0]}"
                else:
                    coordinates_str = " ".join((" ".join(map(str, c)) for c in coords))
                    length = f"{coords.shape[0]},{coords.shape[1]}"

                return f"{values.dtype.name} {length},{values.size} {coordinates_str} {' '.join(map(str, values))}"
            return f"{f.dtype.name} {f.size} {' '.join(map(str, f))}"

        return " ".join(get_f(f) for f in features)

    buffer.write(  # type: ignore
        f"{node_id} -1 {node_type} {node_weight} {_feature_str(node_features)}\n"
    )
    for edge_src, edge_dst, edge_type, edge_weight, edge_features in edges:
        buffer.write(  # type: ignore
            f"{edge_src} {edge_dst} {edge_type} {edge_weight} {_feature_str(edge_features)}\n"
        )


def json_node_to_linear(node, buffer):
    """Convert graph.json to graph.linear."""
    gen = JsonDecoder().decode(node)
    node = next(gen)[1:]
    edges = [edge for edge in gen]
    linear_encode(*node, edges, buffer)


def json_to_linear(filename_in, filename_out):
    """Convert graph.json to graph.linear."""
    file_in = open(filename_in, "r")
    file_out = open(filename_out, "w")
    while True:
        line = file_in.readline().strip()
        if not line:
            break
        json_node_to_linear(line, file_out)
    file_in.close()
    file_out.close()
