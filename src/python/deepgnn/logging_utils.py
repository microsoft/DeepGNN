# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Common logging functions."""

import logging
import logging.config
import os
import socket
import threading
from .log_consts import (
    LOG_PROPS_CUSTOM_DIMENSIONS,
    LOG_PROPS_KEY_EVENT_TYPE,
    LOG_PROPS_KEY_USER_NAME,
    LOG_PROPS_KEY_JOB_ID,
    LOG_PROPS_KEY_WORKER_INDEX,
    LOG_PROPS_KEY_NUM_WORKERS,
    LOG_PROPS_KEY_MODE,
    LOG_PROPS_KEY_MODEL,
    LOG_PROPS_KEY_PLATFORM,
    LOG_NAME_DEEPGNN,
)
from opencensus.ext.azure.log_exporter import AzureLogHandler


_init = False
_logger = None
_logger_lock = threading.Lock()


def get_current_user():
    """Get user name."""
    try:
        return os.getenv("USER", socket.gethostbyname(socket.gethostname()))
    except socket.error:
        return "unknown"


class AzureAppInsightFilter(logging.Filter):
    """AzureAppInsightFilter filter logs which will not send to Azure."""

    # if the log will be sent to azure app insights.
    ENABLE_TELEMETRY = False

    def filter(self, record):
        """Filter records for telemetry."""
        return AzureAppInsightFilter.ENABLE_TELEMETRY and hasattr(
            record, LOG_PROPS_CUSTOM_DIMENSIONS
        )


class DeepGNNAzureLogHandler(AzureLogHandler):
    """Simple handler to log records to azure."""

    def __init__(self, **options):
        """Initialize handler."""
        super(DeepGNNAzureLogHandler, self).__init__(**options)
        self.filters = []


def log_telemetry(
    azlogger,
    content: str,
    key: str,
    mode: str,
    model_name: str,
    user_name: str = "",
    job_id: str = "",
    task_index: int = 0,
    worker_size: int = 1,
    platform: str = "",
):
    """Log training job properties."""
    azlogger.info(
        content,
        extra={
            LOG_PROPS_CUSTOM_DIMENSIONS: {
                LOG_PROPS_KEY_EVENT_TYPE: key,
                LOG_PROPS_KEY_USER_NAME: user_name,
                LOG_PROPS_KEY_JOB_ID: job_id,
                LOG_PROPS_KEY_WORKER_INDEX: task_index,
                LOG_PROPS_KEY_NUM_WORKERS: worker_size,
                LOG_PROPS_KEY_MODE: mode,
                LOG_PROPS_KEY_MODEL: model_name,
                LOG_PROPS_KEY_PLATFORM: platform,
            }
        },
    )


LOGGING = {
    "version": 1,
    "formatters": {
        "detailed": {
            "format": "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "detailed",
            "filters": [],
        }
    },
    "loggers": {
        LOG_NAME_DEEPGNN: {
            "level": "INFO",
            "propagate": False,
            "handlers": ["console"],
        },
        "tensorflow": {"level": "INFO", "propagate": False, "handlers": ["console"]},
    },
}


def add_azure_handler(name: str):
    """Add azure log handler."""
    logger = logging.getLogger(name)
    azure_handler = DeepGNNAzureLogHandler()
    azure_handler.addFilter(AzureAppInsightFilter())
    azure_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(azure_handler)


def setup_default_logging_config(enable_telemetry: bool = False):
    """Create a global logging config."""
    global _init
    if logging.getLoggerClass() is logging.Logger:
        # initialize logging config for default logger
        AzureAppInsightFilter.ENABLE_TELEMETRY = enable_telemetry
        logging.config.dictConfig(LOGGING)
        try:
            add_azure_handler(LOG_NAME_DEEPGNN)
        except ValueError:
            AzureAppInsightFilter.ENABLE_TELEMETRY = False

    _init = True


def get_logger():
    """Initialize logger and then return it."""
    global _logger
    global _init

    if _logger:
        return _logger

    _logger_lock.acquire()

    try:
        if _logger:
            return _logger
        if not _init:
            setup_default_logging_config(False)
        _logger = logging.getLogger(LOG_NAME_DEEPGNN)
        return _logger

    finally:
        _logger_lock.release()
