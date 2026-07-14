"""
app/core/logging.py
====================
Centralized logging configuration for the entire ARAP system.

This module provides a configured logger instance used across all services
(API, Worker, Agents). It formats logs consistently with timestamps,
log levels, module names, and messages.
"""
import logging
import sys

# Define a consistent log format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging(level=logging.INFO):
    """Configure the root logger with a consistent format."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Remove existing handlers to avoid duplicate logs in some environments
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(handler)
    
    # Set specific levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("celery").setLevel(logging.INFO)

# Initialize logging when this module is imported
setup_logging()

# Export a root logger instance for convenience
logger = logging.getLogger(__name__)