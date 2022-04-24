# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
import json
import math
import jinja2
import ansible_runner


def start_distributed_ge(
    data_dir: str,
    port: int,
    deploy_infra: bool = False,
    host_count: int = 1,
    partition_count: int = 1,
):
    """Start distributed graph engine using ansible playbook."""

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    work_dir = os.path.dirname(os.path.abspath(__file__))

    # if force to deploy infrastructure, playbook will remote to
    # each host and install softwares depended.
    if deploy_infra:
        logger.info(f"Start to deploy infrastructure to all machines.")
        r = ansible_runner.run(
            private_data_dir=work_dir, playbook="playbooks/deploy.yml"
        )
        assert r.rc == 0
        return

    # calculate how many hosts will be used to launch the graph engine.
    partition_per_ge = math.ceil(partition_count / host_count)
    inventory_str = ansible_runner.get_inventory(
        action="list", inventories=[os.path.join(work_dir, "inventory")]
    )
    inventory = json.loads(inventory_str[0])
    selected_hosts = []
    for item in inventory["all"]["children"]:
        if item in inventory:
            for i in inventory[item]["hosts"]:
                idx = len(selected_hosts)
                selected_hosts.append(
                    (
                        i,
                        item,
                        ",".join(
                            [
                                str(k)
                                for k in range(
                                    idx * partition_per_ge, (idx + 1) * partition_per_ge
                                )
                            ]
                        ),
                    )
                )
                if len(selected_hosts) >= host_count:
                    break
        if len(selected_hosts) >= host_count:
            break

    logger.info(f"Selected hosts: {selected_hosts}")

    # initialize jinja2 template and render final playbook.
    file_loader = jinja2.FileSystemLoader(os.path.join(work_dir, "playbooks"))
    env = jinja2.Environment(loader=file_loader)
    template = env.get_template("start.yml.template")
    dep_yaml = template.render(data_dir=data_dir, port=port, hosts=selected_hosts)

    tmp_playbook = os.path.join(work_dir, "playbooks/ge_starter.yml")
    with open(tmp_playbook, "w") as f:
        f.write(dep_yaml)

    # start playbook to launch GE.
    r = ansible_runner.run(private_data_dir=work_dir, playbook=tmp_playbook)
    if r.rc != 0:
        logger.error(f"Error when start graph engine in distributed servers.")
    else:
        logger.info(f"Success when start graph engine in distributed servers.")

    os.remove(tmp_playbook)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )

    parser.add_argument(
        "--deploy_infra",
        action="store_true",
        help="Whether or not to deploy the infrastructure to cluster.",
    )

    parser.add_argument("--port", type=str, help="graph engine serivce listening port.")

    parser.add_argument("--host_count", type=int, help="How many hosts will be used.")

    parser.add_argument(
        "--partition_count",
        type=int,
        help="How many partitions in data.",
    )

    parser.add_argument("--data_dir", type=str, help="graph data directory.")

    args = parser.parse_args()
    start_distributed_ge(
        deploy_infra=args.deploy_infra,
        port=args.port,
        host_count=args.host_count,
        partition_count=args.partition_count,
        data_dir=args.data_dir,
    )
