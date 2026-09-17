"""
Unified logging module for AI Service.
Provides centralized logging configuration with file rotation and console output.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


class Logger:
    """Singleton logger wrapper for AI Service."""

    _instance: Optional[logging.Logger] = None
    _initialized: bool = False

    @classmethod
    def initialize(
        cls,
        level: str = "INFO",
        log_dir: str = "logs",
        log_file: str = "ai_service.log",
        max_bytes: int = 10485760,  # 10MB
        backup_count: int = 5,
        console_output: bool = True
    ) -> None:
        """
        Initialize the logger with specified configuration.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory to store log files
            log_file: Log file name
            max_bytes: Maximum size of each log file before rotation
            backup_count: Number of backup files to keep
            console_output: Whether to output logs to console
        """
        if cls._initialized:
            return

        # Create logger
        cls._instance = logging.getLogger("AIService")
        cls._instance.setLevel(getattr(logging, level.upper()))
        cls._instance.handlers.clear()

        # Create log directory if not exists
        os.makedirs(log_dir, exist_ok=True)

        # Formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler with rotation
        log_path = os.path.join(log_dir, log_file)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        cls._instance.addHandler(file_handler)

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, level.upper()))
            console_handler.setFormatter(formatter)
            cls._instance.addHandler(console_handler)

        cls._initialized = True
        # cls._instance.info("Logger initialized successfully")

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """
        Get the logger instance.

        Returns:
            Logger instance

        Raises:
            RuntimeError: If logger not initialized
        """
        if not cls._initialized or cls._instance is None:
            raise RuntimeError("Logger not initialized. Call Logger.initialize() first.")
        return cls._instance


def get_logger() -> logging.Logger:
    """
    Convenience function to get logger instance.

    Returns:
        Logger instance
    """
    return Logger.get_logger()
