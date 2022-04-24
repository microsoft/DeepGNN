# Launch distributed DeepGNN in cluster

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
python distributed.py \
    --port 12345 \
    --data_dir /path/to/your/data/folder \
    --host_count 3 \
    --partition_count 3
```

#### Parameters

name | description |
-----|-------------|
data_dir | the path to the graph data, in distributed mode, this path should be a shared folder/cloud storage or something else which can be accessed by distributed workers. |
port | the graph engine server port |
host_count | how many nodes should be used when lanunching distributed GE, deepgnn will select corresponding hosts instead of using all of them. |
partition_count | how many graph partitions, this parameter will be used to calculate which partitions should be loaded by each worker. |
