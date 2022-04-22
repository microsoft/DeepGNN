# Quick Setup Guide

There are a number of ways to use DeepGNN for experimentation,
* [Azure ML](#azure-ml)
* [Local install for Linux users](#local-install)

## Azure ML

Azure ML is a cloud based end to end machine learning platform. This is a great choice for users working with graph neural networks because it is easy to use and can scale with compute and storage needs.

### Interactive AML

#### Setup workspace

First login to [Azure ML](https://ml.azure.com/). Then either open an existing workspace you have access to or create a new workspace by clicking new, then select the subscription from your org and your region.

Open the 'compute' tab in the last section of the left side menu, if you already have a compute instance or cluster just make sure it is running. Otherwise click the `new` button, if you do not see this you may not have the correct permissions in your current subscription. The simplest machine without a gpu will work to ramp up with. Notably, this compute machine is billed hourly to your org whether you are using it or not so bookmark the `compute` page and stop the machine when you are done with it!

#### Install DeepGNN

We will install DeepGNN through a custom pip feed. In order to use this feed you will need to create a personal access token to our organization with the `Packaging` read permission using the guide, [here](https://docs.microsoft.com/en-us/azure/devops/organizations/accounts/use-personal-access-tokens-to-authenticate?view=azure-devops&tabs=preview-page). If you run into any issues creating or using this token, we would be happy to help at `DeepGNN@microsoft.com`.

Then in the next step you will use the given pip install command per your desired use scenario.

#### Upload Folder

Example folders can be downloaded directly from the [DeepGNN repo examples folder](https://github.com/microsoft/DeepGNN/blob/main/examples), unzipped and uploaded to an AML workspace with the `add files` button on the notebooks tab.

Then in the same tab, open a terminal and paste the following command with ${FEED_ACCESSTOKEN} replaced to install DeepGNN. It only needs to be run once per compute instance and is preserved after shutdown.

```bash
pip install deepgnn-torch --ignore-installed
```

The example code can then be run just the same as it would on your local machine.

#### Upload Notebook

Notebook files can be downloaded directly from the [DeepGNN repo docs folder](https://github.com/microsoft/DeepGNN/blob/main/docs/tutorials) and uploaded to an AML workspace with the `add files` button on the notebooks tab. You can then open the file from the file explorer and run it.

After you upload or create a notebook, paste this pip install command in a cell and run it once with ${FEED_ACCESSTOKEN} replaced. It only needs to be run once per compute instance and is preserved after shutdown, though it is best to run it in the notebook and not a terminal.

```bash
!pip install deepgnn-torch
```

The notebook can then be run in the embedded AML notebook viewer.

### AML Python SDK

(TODO: in deepgraph we use docker to launch this demo, but in snark we use pip package directly, need to have a new notebook without using docker)


### Best use tips

* Do not install large datasets directly to the virtual machine. Instead upload them to the azure storage via the dataset tab from the left menu. Once uploaded, you can load the dataset from the path given. To find the path open the workspaceblobstore (Default) from the dataset tab, click browse on top and select your folder. It'll be in the form `https://<name>.blob.core.windows.net/azureml-blobstore-<int>/azureml/<int>/<folder>`.


## Local Install

A local install is a great option for Linux or WSL users with python >= 3.7. DeepGNN provides 3 separate packages: deepgnn-torch for PyTorch users, deepgnn-tf for Tensorflow users and deepgnn-ge for the graph engine which is required by both frameworks.

```bash
pip install deepgnn-torch
```

```bash
pip install deepgnn-tf
```

Otherwise, DeepGNN can be built on both Ubuntu and WSL(Linux Subsystem for Windows). See the full guide <a href="https://github.com/microsoft/DeepGNN/blob/main/CONTRIBUTING.md" target="_blank">here</a>.

#### <div style="display: inline;float: right">[Next Page](~/onboard/1_node_class.md)</div>
