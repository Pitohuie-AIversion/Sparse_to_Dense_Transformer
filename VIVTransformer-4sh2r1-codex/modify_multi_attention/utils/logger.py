import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logger(
    log_dir: str,
    level: int = logging.INFO,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_name: str = "train.log",
    error_log_name: str = "error.log",
) -> logging.Logger:
    """Sets up a logger with console and file handlers, including a separate error log.

    Args:
        log_dir (str): Directory to save log files.
        level (int, optional): The base logging level for the logger. Defaults to logging.INFO.
        console_level (int, optional): Logging level for the console handler. Defaults to logging.INFO.
        file_level (int, optional): Logging level for the main file handler. Defaults to logging.DEBUG.
        log_name (str, optional): Name of the main log file. Defaults to "train.log".
        error_log_name (str, optional): Name of the error log file. Defaults to "error.log".

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    console_handler.encoding = "utf-8"
    logger.addHandler(console_handler)

    # File Handler for all logs (with rotation)
    log_path = os.path.join(log_dir, log_name)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 10MB per file
    )
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # File Handler for error logs
    error_log_path = os.path.join(log_dir, error_log_name)
    error_handler = logging.FileHandler(error_log_path, encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
    )
    error_handler.setFormatter(error_formatter)
    logger.addHandler(error_handler)

    return logger
