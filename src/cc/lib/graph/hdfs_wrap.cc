// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "hdfs_wrap.h"
#include "types.h"
#include <glog/logging.h>
#include <glog/raw_logging.h>
#include <stdlib.h>

#ifdef SNARK_PLATFORM_LINUX
#include <dlfcn.h>
#include <fcntl.h>

const int32_t BUFFER_SIZE = 4096;

void check_dlsym_error(const char *symbol)
{
    const char *dlsym_error = dlerror();
    if (dlsym_error)
        RAW_LOG_FATAL("Cannot load symbol '%s': %s", symbol, dlsym_error);
}

std::string get_hdfs_home()
{
    std::string hdfs_path = "";

    const char *hdfs_path_ = getenv("HADOOP_HOME");
    if (hdfs_path_ == nullptr)
    {
        RAW_LOG_WARNING("WARNING: HADOOP_HOME not set, may not be able to find libhdfs.so or etc/hadoop/core-site.xml");
    }
    else
    {
        hdfs_path = hdfs_path_;
    }
    auto hdfs_path_size = hdfs_path.size();
    if (hdfs_path_size && hdfs_path[hdfs_path_size - 1] != '/')
        hdfs_path += "/"; // linux only

    return hdfs_path;
}

void parse_hdfs_path(std::string full_path, std::string &data_path, std::string &host, int &port)
{
    data_path = "";
    host = "default";
    port = 0;

    // hdfs://<host/nameservice>:8020/path
    if (!is_hdfs_path(full_path))
    {
        data_path = full_path;
        return;
    }
    else if (full_path.find("file:///") != std::string::npos)
    {
        data_path = full_path;
        host = full_path;
        return;
    }

    int i = 1, j = 1;
    int path_len = full_path.length();
    i = full_path.find('/', i);
    j = full_path.find(':', i);
    if (j == static_cast<int>(std::string::npos))
        j = path_len;
    host = full_path.substr(0, j);
    if (j < path_len)
    {
        i = j;
        j = full_path.find('/', i);
        port = std::stoi(full_path.substr(i + 1, j - i - 1));
    }
    else
    {
        port = 0;
    }
    data_path = full_path;
}

class hdfsBindings
{
  public:
    hdfsBindings()
    {
        std::string hdfs_path = get_hdfs_home();
        const char *ld_library_path = getenv("LD_LIBRARY_PATH");
        if (ld_library_path == nullptr)
            RAW_LOG_WARNING("WARNING: LD_LIBRARY_PATH not set, may not be able to find libjvm.so");
        const char *java_path_ = getenv("JAVA_HOME");
        std::string java_path = "";
        if (java_path_ != nullptr)
            java_path = java_path_;
        std::string ld_library_path_str = "";
        if (ld_library_path != nullptr)
            ld_library_path_str = ld_library_path;

        std::string libhdfs_path = hdfs_path + std::string("lib/native/libhdfs.so");
        hdfs_handle = dlopen(libhdfs_path.c_str(), RTLD_NOW);
        if (!hdfs_handle)
            RAW_LOG_FATAL("Cannot open library: %s", dlerror());
        dlerror(); // Reset errors

        hdfsConnect_so = (hdfsConnect_t)(dlsym(hdfs_handle, "hdfsConnect"));
        check_dlsym_error("hdfsConnect");
        hdfsExists_so = (hdfsExists_t)(dlsym(hdfs_handle, "hdfsExists"));
        check_dlsym_error("hdfsExists");
        hdfsGetPathInfo_so = (hdfsGetPathInfo_t)(dlsym(hdfs_handle, "hdfsGetPathInfo"));
        check_dlsym_error("hdfsGetPathInfo");
        hdfsOpenFile_so = (hdfsOpenFile_t)dlsym(hdfs_handle, "hdfsOpenFile");
        check_dlsym_error("hdfsOpenFileInfo");
        hdfsFreeFileInfo_so = reinterpret_cast<hdfsFreeFileInfo_t>(dlsym(hdfs_handle, "hdfsFreeFileInfo"));
        check_dlsym_error("hdfsFreeFileInfo");
        hdfsRead_so = reinterpret_cast<hdfsRead_t>(dlsym(hdfs_handle, "hdfsRead"));
        check_dlsym_error("hdfsRead");
        hdfsCloseFile_so = reinterpret_cast<hdfsCloseFile_t>(dlsym(hdfs_handle, "hdfsCloseFile"));
        check_dlsym_error("hdfsCloseFile");
        hdfsDisconnect_so = reinterpret_cast<hdfsDisconnect_t>(dlsym(hdfs_handle, "hdfsDisconnect"));
        check_dlsym_error("hdfsDisconect");
        hdfsListDirectory_so = reinterpret_cast<hdfsListDirectory_t>(dlsym(hdfs_handle, "hdfsListDirectory"));
        check_dlsym_error("hdfsListDirectory");
    }

    ~hdfsBindings()
    {
        dlclose(hdfs_handle);
    }

    typedef hdfsFS_so (*hdfsConnect_t)(const char *, uint16_t);
    typedef hdfs_int (*hdfsExists_t)(hdfsFS_so, const char *);
    typedef hdfsFileInfo_so *(*hdfsGetPathInfo_t)(hdfsFS_so, const char *);
    typedef hdfs_int (*hdfsFreeFileInfo_t)(hdfsFileInfo_so *, hdfs_int);
    typedef hdfsFile_so (*hdfsOpenFile_t)(hdfsFS_so, const char *, hdfs_int, hdfs_int, short, int32_t);
    typedef int32_t (*hdfsRead_t)(hdfsFS_so, hdfsFile_so, void *, int32_t);
    typedef void (*hdfsCloseFile_t)(hdfsFS_so, hdfsFile_so);
    typedef hdfs_int (*hdfsDisconnect_t)(hdfsFS_so);
    typedef hdfsFileInfo_so *(*hdfsListDirectory_t)(hdfsFS_so, const char *, hdfs_int *);

    hdfsConnect_t hdfsConnect_so = nullptr;
    hdfsExists_t hdfsExists_so = nullptr;
    hdfsGetPathInfo_t hdfsGetPathInfo_so = nullptr;
    hdfsFreeFileInfo_t hdfsFreeFileInfo_so = nullptr;
    hdfsOpenFile_t hdfsOpenFile_so = nullptr;
    hdfsRead_t hdfsRead_so = nullptr;
    hdfsCloseFile_t hdfsCloseFile_so = nullptr;
    hdfsDisconnect_t hdfsDisconnect_so = nullptr;
    hdfsListDirectory_t hdfsListDirectory_so = nullptr;

  private:
    void *hdfs_handle = nullptr;
};

HDFSConnection::HDFSConnection()
{
}

HDFSConnection::HDFSConnection(std::string data_path, std::string config_path)
{
    if (config_path != "" && !std::filesystem::exists(config_path))
        RAW_LOG_FATAL("HDFS config path does not exist: %s", config_path.c_str());

    hdfs_bindings = std::make_shared<hdfsBindings>();

    std::string path;
    std::string host_str;
    int port;
    parse_hdfs_path(data_path, path, host_str, port);
    const char *host = host_str.c_str();

    fs = hdfs_bindings->hdfsConnect_so(host, port);
    if (fs == nullptr)
        RAW_LOG_FATAL("HDFS failed to connect!");

    m_data_path = data_path;
}

int64_t HDFSConnection::get_file_size(const char *path, const char *host, int port)
{
    if (fs == nullptr)
        RAW_LOG_FATAL("HDFS unexpectedly disconnected!");
    if (hdfs_bindings->hdfsExists_so(fs, path) != 0)
        RAW_LOG_FATAL("File does not exist '%s' error with: %s", path, strerror(errno));
    auto file_info = hdfs_bindings->hdfsGetPathInfo_so(fs, path);
    auto read_size = (file_info->mSize);
    hdfs_bindings->hdfsFreeFileInfo_so(file_info, 1);
    return read_size;
}

std::vector<std::string> HDFSConnection::list_directory(const char *full_path)
{
    int n_items = 1;
    auto directory_infos = hdfs_bindings->hdfsListDirectory_so(fs, full_path, &n_items);

    std::vector<std::string> output;
    for (int i = 0; i < n_items; i++)
        output.push_back(directory_infos[i].mName);

    hdfs_bindings->hdfsFreeFileInfo_so(directory_infos, n_items);
    return output;
}

hdfsFile_so HDFSConnection::open_file(const char *path)
{
    auto readFile = hdfs_bindings->hdfsOpenFile_so(fs, path, O_RDONLY, 0, 0, 0);
    if (!readFile)
        RAW_LOG_FATAL("Failed to open %s for reading", path);

    m_buffer = malloc(BUFFER_SIZE);

    return readFile;
}

void HDFSConnection::close_file(hdfsFile_so readFile)
{
    hdfs_bindings->hdfsCloseFile_so(fs, readFile);
    free(m_buffer);
    m_buffer = nullptr;
}

void HDFSConnection::read(hdfsFile_so readFile, int64_t read_size, void *output)
{
    if (readFile == nullptr)
        RAW_LOG_FATAL("Read input file not open!");

    uint8_t *src = static_cast<uint8_t *>(m_buffer);
    uint8_t *dst = static_cast<uint8_t *>(output);

    int32_t read_size_curr;
    int32_t curr_read = 0;
    int64_t acc_read = 0;
    while (read_size > acc_read)
    {
        read_size_curr = static_cast<int32_t>(std::min(read_size - acc_read, static_cast<int64_t>(BUFFER_SIZE)));
        curr_read = hdfs_bindings->hdfsRead_so(fs, readFile, m_buffer, read_size_curr);
        if (curr_read < read_size_curr)
            curr_read +=
                hdfs_bindings->hdfsRead_so(fs, readFile, static_cast<void *>(static_cast<char *>(m_buffer) + curr_read),
                                           read_size_curr - curr_read);
        if (curr_read != read_size_curr)
            RAW_LOG_FATAL("file %s : Stopped reading after %li bytes, expected %li bytes!", m_data_path.c_str(),
                          acc_read, read_size);
        memcpy(dst, src, read_size_curr);
        dst += curr_read;
        acc_read += curr_read;
    }
}

std::vector<std::string> hdfs_list_directory(std::string full_path, std::string config_path)
{
    std::string data_path_str;
    std::string host_str;
    int port;
    parse_hdfs_path(full_path, data_path_str, host_str, port);
    const char *data_path = data_path_str.c_str();

    auto connection = HDFSConnection(full_path, config_path);
    return connection.list_directory(data_path);
}

template <typename T> std::vector<T> read_hdfs(std::string full_path, std::string config_path)
{
    std::string data_path_str;
    std::string host_str;
    int port;
    parse_hdfs_path(full_path, data_path_str, host_str, port);
    const char *data_path = data_path_str.c_str();
    const char *host = host_str.c_str();

    auto connection = HDFSConnection(full_path, config_path);
    auto read_size = connection.get_file_size(data_path, host, port);
    std::vector<T> output(read_size / sizeof(T));
    auto file = connection.open_file(data_path);
    connection.read(file, read_size, static_cast<void *>(output.data()));
    connection.close_file(file);
    return output;
}

#else

HDFSConnection::HDFSConnection()
{
}

HDFSConnection::HDFSConnection(std::string data_path, std::string config_path)
{
    m_data_path = data_path;
    hdfs_bindings = nullptr;
    fs = nullptr;
    m_buffer = nullptr;
}

int64_t HDFSConnection::get_file_size(const char *path, const char *host, int port)
{
    return -1;
}

std::vector<std::string> HDFSConnection::list_directory(const char *full_path)
{
    return std::vector<std::string>();
}

hdfsFile_so HDFSConnection::open_file(const char *path)
{
    return nullptr;
}

void HDFSConnection::close_file(hdfsFile_so readFile)
{
}

void HDFSConnection::read(hdfsFile_so readFile, int64_t read_size, void *output)
{
}

void parse_hdfs_path(std::string full_path, std::string &data_path, std::string &host, int &port)
{
}

std::vector<std::string> hdfs_list_directory(std::string full_path, std::string config_path)
{
    RAW_LOG_FATAL("HDFS only supported for linux!");
    return std::vector<std::string>();
}

template <typename T> std::vector<T> read_hdfs(std::string full_path, std::string config_path)
{
    RAW_LOG_FATAL("HDFS only supported for linux!");
    return std::vector<T>();
}

#endif

bool is_hdfs_path(std::filesystem::path path)
{
    return path.string().find("adl://") == 0 || path.string().find("hdfs://") == 0 ||
           path.string().find("file:///") == 0;
}

template std::vector<uint8_t> read_hdfs(std::string full_path, std::string config_path);
template std::vector<uint16_t> read_hdfs(std::string full_path, std::string config_path);
template std::vector<uint32_t> read_hdfs(std::string full_path, std::string config_path);
template std::vector<uint64_t> read_hdfs(std::string full_path, std::string config_path);
template std::vector<char> read_hdfs(std::string full_path, std::string config_path);
