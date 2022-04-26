# HDFS Streaming

DeepGNN on Linux supports direct HDFS/ADL stream to memory. In order to use this feature, you must have hadoop installed, a few environment variables set and GE options set.

## Hadoop Download

### Pip Install

Follow the Hadoop install guide, [here](https://hadoop.apache.org/docs/r3.3.2/hadoop-project-dist/hadoop-common/SingleCluster.html#Installing_Software). Make sure to verify the CLI works with the command they give before continuing.

### Build from source

If you build DeepGNN from source with bazel, you can use the following target to download HDFS,

```bash
bazel test //src/cc/tests:hdfs_tests --config=linux
```

## Environment Variables

```bash
export HADOOP_HOME=/path/to/hadoop

# If building from source using bazel, keep empty and set this value in code instead
export JAVA_HOME=/path/to/java

# Only enter if building from source or you manually download java jdk
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$JAVA_HOME/jre/lib/amd64/server/"
```

If CLASSPATH is not already set, it will be set automatically with config_path at the top.

## cores-site.xml

A core-site.xml is the main configuration file for hadoop. Below are some quick examples that can be copy and pasted.

You can test core-site files with

```bash
echo 'export HADOOP_CLASSPATH=$HADOOP_HOME/share/hadoop/tools/lib/*' >> etc/hadoop/hadoop-env.sh
sudo bin/hdfs dfs --conf core-site.xml -ls <HDFS_PATH>
```

ADL Example

```xml
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <property>
        <name>fs.adl.impl</name>
        <value>org.apache.hadoop.fs.adl.AdlFileSystem</value>
    </property>
    <property>
        <name>fs.adl.oauth2.refresh.url</name>
        <value>https://login.microsoftonline.com/TODO_TENANT_ID/oauth2/token</value>
    </property>
    <property>
        <name>fs.adl.oauth2.access.token.provider.type</name>
        <value>ClientCredential</value>
    </property>
    <property>
        <name>fs.adl.oauth2.client.id</name>
        <value>TODO_CLIENT_ID</value>
    </property>
    <property>
        <name>fs.adl.oauth2.credential</name>
        <value>TODO_PASSWORD</value>
    </property>
    <property>
        <name>io.file.buffer.size</name>
        <value>4194304</value>
    </property>
    <property>
        <name>fs.parallel-copy.use</name>
        <value>true</value>
    </property>
    <property>
        <name>fs.parallel-copy.detect.text</name>
        <value>true</value>
    </property>
    <property>
        <name>fs.parallel-copy.text-file.scope-compatible</name>
        <value>true</value>
    </property>
    <property>
        <name>fs.permissions.umask-mode</name>
        <value>002</value>
    </property>
</configuration>
```

HDFS Localhost Example

```xml
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://localhost:9000</value>
    </property>
</configuration>
```

## Graph Engine Usage

Leverage this feature by setting --data_dir to an hdfs or adl link, adding --stream and --config_path path/to/core-site.xml.
