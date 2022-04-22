// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_HDFS_WRAP_H
#define SNARK_HDFS_WRAP_H

#include <filesystem>
#include <vector>

typedef int32_t hdfs_int;

struct hdfs_internal_so;
typedef struct hdfs_internal_so *hdfsFS_so;

enum hdfsStreamType_so
{
    HDFS_STREAM_UNINITIALIZED = 0,
    HDFS_STREAM_INPUT = 1,
    HDFS_STREAM_OUTPUT = 2,
};
struct hdfsFile_internal_so
{
    void *file;
    enum hdfsStreamType_so type;
    hdfs_int flags;
};
typedef hdfsFile_internal_so *hdfsFile_so;

typedef enum tObjectKind_so
{
    kObjectKindFile_so = 'F',
    kObjectKindDirectory_so = 'D',
} tObjectKind_so;

typedef struct
{
    tObjectKind_so mKind; /* file or directory */
    char *mName;          /* the name of the file */
    time_t mLastMod;      /* the last modification time for the file in seconds */
    int64_t mSize;        /* the size of the file in bytes */
    short mReplication;   /* the count of replicas */
    int64_t mBlockSize;   /* the block size for the file */
    char *mOwner;         /* the owner of the file */
    char *mGroup;         /* the group associated with the file */
    short mPermissions;   /* the permissions associated with the file */
    time_t mLastAccess;   /* the last access time for the file in seconds */
} hdfsFileInfo_so;

class hdfsBindings;

class HDFSConnection
{
  public:
    HDFSConnection();
    HDFSConnection(std::string data_path, std::string config_path);

    int64_t get_file_size(const char *path, const char *host, int port);

    std::vector<std::string> list_directory(const char *full_path);

    hdfsFile_so open_file(const char *path);
    void close_file(hdfsFile_so readFile);
    void read(hdfsFile_so readFile, int64_t read_size, void *output);

  private:
    std::shared_ptr<hdfsBindings> hdfs_bindings;
    hdfsFS_so fs = nullptr;
    std::string m_data_path = "";
    void *m_buffer = nullptr;
};

void parse_hdfs_path(std::string full_path, std::string &data_path, std::string &host, int &port);

std::vector<std::string> hdfs_list_directory(std::string full_path, std::string config_path);

template <typename T> std::vector<T> read_hdfs(std::string full_path, std::string config_path);

bool is_hdfs_path(std::filesystem::path path);

#endif
