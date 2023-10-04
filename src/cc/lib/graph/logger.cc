#include "src/cc/lib/graph/logger.h"

#include <cstdarg>
// Use raw log to avoid possible initialization conflicts with glog from other libraries.
#include <glog/logging.h>
#include <glog/raw_logging.h>

namespace snark
{

void GLogger::log_info(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    std::string msg;
    char buffer[256];
#ifdef _WIN32
    vsnprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1, format, args);
#else
    vsnprintf(buffer, sizeof(buffer), format, args);
#endif
    va_end(args);
    msg = buffer;
    RAW_LOG_INFO("%s", msg.c_str());
}

void GLogger::log_error(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    std::string msg;
    char buffer[256];
#ifdef _WIN32
    vsnprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1, format, args);
#else
    vsnprintf(buffer, sizeof(buffer), format, args);
#endif
    va_end(args);
    msg = buffer;
    RAW_LOG_ERROR("%s", msg.c_str());
}

void GLogger::log_warning(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    std::string msg;
    char buffer[256];
#ifdef _WIN32
    vsnprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1, format, args);
#else
    vsnprintf(buffer, sizeof(buffer), format, args);
#endif
    va_end(args);
    msg = buffer;
    RAW_LOG_WARNING("%s", msg.c_str());
}

void GLogger::log_fatal(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    std::string msg;
    char buffer[256];
#ifdef _WIN32
    vsnprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1, format, args);
#else
    vsnprintf(buffer, sizeof(buffer), format, args);
#endif
    va_end(args);
    msg = buffer;
    RAW_LOG_FATAL("%s", msg.c_str());
}

} // namespace snark
