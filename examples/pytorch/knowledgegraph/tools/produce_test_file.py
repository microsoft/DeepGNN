# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


def main(root_path: str):
    entities_path = root_path + "entities.dict"
    relations_path = root_path + "relations.dict"
    test_path = root_path + "test.txt"

    entities_dict = {}
    relations_dict = {}

    with open(entities_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            cols = line.split("\t")
            entities_dict[cols[1]] = cols[0]

    with open(relations_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            cols = line.split("\t")
            relations_dict[cols[1]] = cols[0]
    output_path = root_path + "test_edge_t.txt"

    with open(test_path, "r") as f:
        lines = f.readlines()
        # head batch
        mode = "0"
        output_file = open(output_path, "w+")
        for line in lines:
            line = line.rstrip()
            cols = line.split("\t")
            src_id = entities_dict[cols[0]]
            dst_id = entities_dict[cols[2]]
            rlt_id = relations_dict[cols[1]]
            output_file.write("\t".join([src_id, dst_id, rlt_id, mode]))
        mode = "1"
        for line in lines:
            line = line.rstrip()
            cols = line.split("\t")
            src_id = entities_dict[cols[0]]
            dst_id = entities_dict[cols[2]]
            rlt_id = relations_dict[cols[1]]
            output_file.write("\t".join([src_id, dst_id, rlt_id, mode]))

        output_file.close()


if __name__ == "__main__":
    root_path = "data/rotatE/FB15k/"
    main(root_path)
