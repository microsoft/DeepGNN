# Launch distributed DeepGNN in cluster

***Note:*** This example is used to setup DeepGNN in distributed machines with big dataset. When processing big dataset, we need to convert the graph into several partitions and each partition should be loaded by a graph engine process. So we need to setup DeepGNN enviroment in these distributed machines. This example uses ansible to remote to each machine and install DeeGNN and its dependencies in the machine. It will also use a jinja2 template to generate the command to launch graph engine process in each machine.

`./ansible/inventory/hosts`: Ansible hosts files which contains all the agents which are used to launch distributed graph engine services (linux and windows).
`./ansible/playbooks/deploy.yml`: Ansible playbook to install DeepGNN and its depedencies in each agent (from hosts file).
`./ansible/playbooks/centos.yml`: Some special commands to install DeepGNN dependencies in CentOS . This is used by `./ansible/playbooks/deploy.yml`.
`./ansible/playbooks/ubuntu.yml`: Some special commands to install DeepGNN dependencies in Ubuntu. This is used by `./ansible/playbooks/deploy.yml`.
`./ansible/playbooks/start.yml.template`: A jinja2 template which is used to generate commands to launch distributed graph engine servers. This template will be rendered using hosts from `./ansible/inventory/hosts`
`./ansible/distribute.py`: A python entrance to launch ansible scripts above.

## Using ansible

### Preparation

1. Linux nodes

    a) Setup SSH: https://docs.microsoft.com/en-us/azure/virtual-machines/linux/ssh-from-windows

    b) Go to `Inventory` folder and add your linux servers in `linux` section:

    ```YAML
    linux:
      hosts:
        azure-linux-vm01: # machine alias
          ansible_host: x.x.x.x # machine ip
      vars:
        ansible_connection: paramiko
        ansible_user: azureuser # ssh user
        ansible_ssh_pass: "password" # ssh password
    ```

2. Windows nodes

    a) Setup windows powershell remote: https://docs.ansible.com/ansible/latest/user_guide/windows.html

    b) Go to `Inventory` folder and add your linux servers in `windows` section:

    ```YAML
    windows:
      hosts:
        azure-win-vm02: # machine alias
          ansible_host: x.x.x.x # machine ip
      vars:
        ansible_connection: psrp
        ansible_psrp_protocol: http
        ansible_port: 5985
        ansible_user: azureuser # windows user
        ansible_password: "password"
        ansible_python_interpreter: python
    ```

### Usage

Go to `distributed/ansible` folder and invoke following command:

```Shell
python ./distribute.py --deploy_infra
python -m deepgnn.graph_engine.data.citation --data_dir /tmp/cora
python ./distribute.py --port 12345 --data_dir /tmp/cora --host_count 1 --partition_count 1
```

#### Parameters

name | description |
-----|-------------|
data_dir | the path to the graph data, in distributed mode, this path should be a shared folder/cloud storage or something else which can be accessed by distributed workers. |
port | the graph engine server port |
host_count | how many nodes should be used when lanunching distributed GE, deepgnn will select corresponding hosts instead of using all of them. |
partition_count | how many graph partitions, this parameter will be used to calculate which partitions should be loaded by each worker. |
