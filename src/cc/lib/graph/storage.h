// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_STORAGE_BASE_H
#define SNARK_STORAGE_BASE_H

#include <cassert>
#include <cstdio>
#include <iterator>
#include <span>
#include <stdexcept>

#include "locator.h"
#include "logger.h"
#include "types.h"

namespace
{
typedef FILE *(*open_file_ptr)(std::filesystem::path, std::string, std::shared_ptr<snark::Logger>);
typedef FILE *(*open_alias_file_ptr)(std::filesystem::path, size_t, snark::Type, std::shared_ptr<snark::Logger>);

} // namespace

namespace snark
{
struct FilePtr
{
  public:
    explicit FilePtr(FILE *file_ptr)
    {
        m_file_ptr = file_ptr;
    }
    ~FilePtr()
    {
        if (m_file_ptr != nullptr)
            fclose(m_file_ptr);
    }

    FILE *operator*()
    {
        return m_file_ptr;
    }

  private:
    FILE *m_file_ptr = nullptr;
};

template <typename T> struct BaseStorage
{
  public:
    virtual ~BaseStorage() = default;
    virtual size_t size() = 0;
    virtual std::shared_ptr<FilePtr> start() = 0;
    virtual size_t read(void *output, size_t size, size_t count, std::shared_ptr<FilePtr> file_ptr_temp) = 0;
    virtual typename std::span<T>::iterator read(uint64_t offset, uint64_t size,
                                                 typename std::span<T>::iterator output_ptr,
                                                 std::shared_ptr<FilePtr> file_ptr) const = 0;
    virtual size_t write(uint64_t offset, uint64_t size, typename std::span<const T>::iterator input_ptr,
                         std::shared_ptr<FilePtr> file_ptr) = 0;
};

template <typename T> struct MemoryStorage : BaseStorage<T>
{
  public:
    MemoryStorage()
    {
    }
    MemoryStorage(std::vector<T> data)
    {
        m_data = std::move(data);
    }
    MemoryStorage(const std::filesystem::path path, const std::string suffix, open_file_ptr open_file,
                  std::shared_ptr<Logger> logger)
    {
        if (open_file == nullptr)
            return;
        if (!logger)
        {
            logger = std::make_shared<GLogger>();
        }
        m_logger = logger;

        auto file_ptr = open_file(std::move(path), std::move(suffix), m_logger);

        snark::platform_fseek(file_ptr, 0L, SEEK_END);
        auto size = snark::platform_ftell(file_ptr);
        m_data.resize(size);

        snark::platform_fseek(file_ptr, 0L, SEEK_SET);
        assert(size == fread(m_data.data(), 1, size, file_ptr) && "Failed to read node features data");

        fclose(file_ptr);
    }

    size_t size() override
    {
        return m_data.size();
    }

    std::shared_ptr<FilePtr> start() override
    {
        return std::make_shared<FilePtr>(nullptr);
    }

    size_t read(void *output, size_t size, size_t count, std::shared_ptr<FilePtr> file_ptr_temp) override
    {
        assert(false && "pointer read not supported by MemoryStorage!");
        return -1;
    }

    typename std::span<T>::iterator read(uint64_t offset, uint64_t size, typename std::span<T>::iterator output_ptr,
                                         std::shared_ptr<FilePtr> file_ptr) const override
    {
        std::copy_n(std::begin(m_data) + offset, sizeof(T) * size, output_ptr);
        output_ptr += size;
        return output_ptr;
    }

    size_t write(uint64_t offset, uint64_t size, typename std::span<const T>::iterator input_ptr,
                 std::shared_ptr<FilePtr> file_ptr) override
    {
        std::copy_n(input_ptr, sizeof(T) * size, std::begin(m_data) + offset);
        return sizeof(T) * size;
    }

  private:
    std::vector<T> m_data;
    std::shared_ptr<Logger> m_logger;
};

template <typename T> struct HDFSStorage final : public MemoryStorage<T>
{
  public:
    HDFSStorage(const char *hdfs_path, const std::string &config_path)
        : MemoryStorage<T>(std::move(read_hdfs<T>(hdfs_path, config_path)))
    {
    }

    HDFSStorage(const wchar_t *hdfs_path, const std::string &config_path)
    {
        assert(false && "HDFS only supported on linux!");
    }
};

template <typename T> struct HDFSStreamStorage final : BaseStorage<T>
{
  public:
    HDFSStreamStorage(const char *hdfs_path, const std::string &config_path)
    {
        std::string data_path_str;
        std::string host_str;
        int port;
        parse_hdfs_path(hdfs_path, data_path_str, host_str, port);
        const char *data_path = data_path_str.c_str();
        const char *host = host_str.c_str();

        m_connection = HDFSConnection(std::string(hdfs_path), config_path);
        m_size = m_connection.get_file_size(data_path, host, port);

        m_file_ptr = m_connection.open_file(data_path);

        m_buffer = static_cast<char *>(malloc(BUFFER_SIZE));
    }

    HDFSStreamStorage(const wchar_t *hdfs_path, const std::string &config_path)
    {
        assert(false && "HDFS only supported on linux!");
    }

    ~HDFSStreamStorage()
    {
        free(m_buffer);
        if (m_file_ptr != nullptr)
        {
            m_connection.close_file(m_file_ptr);
        }
    }

    size_t size() override
    {
        return m_size;
    }

    std::shared_ptr<FilePtr> start() override
    {
        m_buffer_offset = BUFFER_SIZE;
        m_offset = 0;
        return std::make_shared<FilePtr>(nullptr);
    }

    size_t read(void *output, size_t size, size_t count, std::shared_ptr<FilePtr> file_ptr_temp) override
    {
        size_t bytes = size * count;
        size_t buffer_left = BUFFER_SIZE - m_buffer_offset;

        if (bytes > static_cast<size_t>(BUFFER_SIZE))
        {
            memcpy(output, m_buffer + m_buffer_offset, buffer_left);
            m_connection.read(m_file_ptr, bytes - buffer_left, static_cast<char *>(output) + buffer_left);
            m_buffer_offset = BUFFER_SIZE;
            m_offset += bytes - buffer_left;
            return count;
        }

        if (m_offset == m_size && buffer_left < bytes)
            throw std::out_of_range("Offset out of range!");

        if (buffer_left < bytes)
        {
            memcpy(m_buffer, m_buffer + m_buffer_offset, buffer_left);
            int64_t read_size = std::min(m_buffer_offset, m_size - m_offset);

            if (read_size <= 0)
                throw std::out_of_range("File closed unexpectedly!");

            m_connection.read(m_file_ptr, read_size, m_buffer + buffer_left);
            m_buffer_offset = 0;
            m_offset += read_size;
        }

        memcpy(output, m_buffer + m_buffer_offset, bytes);
        m_buffer_offset += bytes;
        return count;
    }

    typename std::span<T>::iterator read(uint64_t offset, uint64_t size, typename std::span<T>::iterator output_ptr,
                                         std::shared_ptr<FilePtr> file_ptr) const override
    {
        assert(false && "HDFSStreamStorage does not support iterator read!");
        return output_ptr;
    }

    size_t write(uint64_t offset, uint64_t size, typename std::span<const T>::iterator input_ptr,
                 std::shared_ptr<FilePtr> file_ptr) override
    {
        assert(false && "HDFSStreamStorage does not support writes!");
        return 0;
    }

  private:
    const int64_t BUFFER_SIZE = 4096;
    HDFSConnection m_connection;
    hdfsFile_so m_file_ptr = nullptr;
    size_t m_size;

    char *m_buffer;
    size_t m_buffer_offset;
    size_t m_offset;
    std::shared_ptr<Logger> m_logger;
};

template <typename T> struct DiskStorage final : BaseStorage<T>
{
  public:
    DiskStorage(std::filesystem::path path, std::string suffix, open_file_ptr open_file, std::shared_ptr<Logger> logger)
    {
        if (!logger)
        {
            logger = std::make_shared<GLogger>();
        }
        m_logger = logger;
        m_path = std::move(path);
        m_suffix = std::move(suffix);
        m_open_file = open_file;
        if (open_file == nullptr)
            return;

        auto file_ptr = m_open_file(m_path, m_suffix, m_logger);

        snark::platform_fseek(file_ptr, 0L, SEEK_END);
        m_size = snark::platform_ftell(file_ptr);

        fclose(file_ptr);
    }

    DiskStorage(std::filesystem::path path, size_t partition, snark::Type type, open_alias_file_ptr open_file,
                std::shared_ptr<Logger> logger)
    {
        if (!logger)
        {
            logger = std::make_shared<GLogger>();
        }
        m_logger = logger;

        m_path = std::move(path);
        m_partition = partition;
        m_type = type;
        m_open_alias_file = open_file;
        if (open_file == nullptr)
            return;

        auto file_ptr = m_open_alias_file(m_path, m_partition, m_type, m_logger);

        snark::platform_fseek(file_ptr, 0L, SEEK_END);
        m_size = snark::platform_ftell(file_ptr);
        fclose(file_ptr);
    }

    size_t size() override
    {
        return m_size;
    }

    std::shared_ptr<FilePtr> start() override
    {
        FILE *file_ptr;
        if (m_open_file != nullptr)
            file_ptr = m_open_file(m_path, m_suffix, m_logger);
        else
            file_ptr = m_open_alias_file(m_path, m_partition, m_type, m_logger);

        return std::make_shared<FilePtr>(file_ptr);
    }

    size_t read(void *output, size_t size, size_t count, std::shared_ptr<FilePtr> file_ptr_temp) override
    {
        FILE *file_ptr = **file_ptr_temp;
        if (file_ptr == nullptr)
            throw std::out_of_range("File not open!");
        if (feof(file_ptr))
            throw std::out_of_range("File closed unexpectedly!");

        return fread(output, size, count, file_ptr);
    }

    typename std::span<T>::iterator read(uint64_t offset, uint64_t size, typename std::span<T>::iterator output_ptr,
                                         std::shared_ptr<FilePtr> file_ptr_temp) const override
    {
        FILE *file_ptr = **file_ptr_temp;

        if (file_ptr == nullptr)
            throw std::out_of_range("File not open!");
        if (feof(file_ptr))
            throw std::out_of_range("File closed unexpectedly!");
        if (offset >= m_size)
            throw std::out_of_range("Offset out of range!");

        snark::platform_fseek(file_ptr, offset, SEEK_SET);

        output_ptr += fread(&(*output_ptr), sizeof(T), size, file_ptr);

        return output_ptr;
    }

    size_t write(uint64_t offset, uint64_t size, typename std::span<const T>::iterator input_ptr,
                 std::shared_ptr<FilePtr> file_ptr) override
    {
        assert(false && "Disk storage does not support writes!");
        return 0;
    }

  private:
    std::filesystem::path m_path = "";
    std::string m_suffix = "";
    size_t m_partition = 0;
    snark::Type m_type = 0;
    open_file_ptr m_open_file = nullptr;
    open_alias_file_ptr m_open_alias_file = nullptr;
    uint64_t m_size = 0;
    std::shared_ptr<Logger> m_logger;
};

#endif

} // namespace snark
