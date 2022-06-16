# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Type conversions to parse command line arguments."""
import argparse


def vec2str(vec):
    """Concat 1 or 2 dimensionanl ndarray to str.

    It is used when output embeddings to files, e.g.:
    vec = [1,2,3]
    res = vec2str(vec)
    # res is "1 2 3"
    """
    if len(vec.shape) == 2:
        return ",".join([" ".join([str(i) for i in j]) for j in vec.tolist()])
    elif len(vec.shape) == 1:
        return " ".join([str(i) for i in vec.tolist()])
    else:
        raise RuntimeError("wrong shape: " + str(vec.shape))


def str2bool(v):
    """Convert the string value to bool.

    For example:
    str2bool("yes") # True
    str2bool("Yes") # True
    str2bool("True") # True
    str2bool("true") # True
    str2bool("False") # False
    str2bool("0") # False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# str to 1d list
def str2list_int(v):
    """Convert a comma separated string to int list.

    For example:
    str2list_int("1,2,3") # result is [1,2,3]
    str2list_int([1,2,3]) # result is [1,2,3]
    """
    if isinstance(v, list):
        return v
    if v == "":
        return []
    return [int(x) for x in v.split(",")]


# str to 2d int list
def str2list2_int(v):
    """Convert string to 2d-list.

    It is used to parse the metapath of the node.
    metapath = "1;2;3,4"
    str2list2_int(metapath) # [[1],[2],[3, 4]]
    """
    if isinstance(v, list):
        return v
    ret = []
    for y in v.split(";"):
        if y != "":
            ret.append([int(x) for x in y.split(",")])
    return ret


# str to 2d list
def str2list2(v):
    """Convert string to string list.

    It is used to parse the edge types of the node.
    edges = "q;k;s"
    str2list2(edges) # [['q'],['k'],['s']]
    """
    if isinstance(v, list):
        return v
    ret = []
    for y in v.split(";"):
        if y != "":
            ret.append([x for x in y.split(",")])
    return ret


#
def str2list_str(v):
    """Convert string to string list, default separator is ","."""
    return v.split(",")
