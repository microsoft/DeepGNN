// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_LOGGER_H
#define SNARK_LOGGER_H

#include <string>

namespace snark
{

// Simple logger interface to allow non glog logging.
struct Logger
{
    virtual void log_info(const char *format, ...) = 0;
    virtual void log_error(const char *format, ...) = 0;
    virtual void log_warning(const char *format, ...) = 0;
    virtual void log_fatal(const char *format, ...) = 0;
    virtual ~Logger() = default;
};

// Logger implementation that uses glog.
struct GLogger : public Logger
{
    void log_info(const char *format, ...) override;
    void log_error(const char *format, ...) override;
    void log_warning(const char *format, ...) override;
    void log_fatal(const char *format, ...) override;
};

} // namespace snark

#endif // SNARK_LOGGER_H
