<<<<<<< HEAD
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

sudo apt-get install openjdk-8-jre
export HADOOP_VERSION=hadoop-3.3.1
wget -nc https://dlcdn.apache.org/hadoop/common/$HADOOP_VERSION/$HADOOP_VERSION.tar.gz
tar -xvzf $HADOOP_VERSION.tar.gz
export HADOOP_HOME=$(pwd)/$HADOOP_VERSION
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$JAVA_HOME/jre/lib/amd64/server/"
=======
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

sudo apt-get update
sudo apt-get install openjdk-8-jre
export HADOOP_VERSION=hadoop-3.3.1
wget -nc https://dlcdn.apache.org/hadoop/common/$HADOOP_VERSION/$HADOOP_VERSION.tar.gz
tar -xvzf $HADOOP_VERSION.tar.gz
export HADOOP_HOME=$(pwd)/$HADOOP_VERSION
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$JAVA_HOME/jre/lib/amd64/server/"
>>>>>>> b30762e9d10d2ae19e206c7622f6d10554dc84f0
