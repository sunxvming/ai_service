"""
Configuration loader module for AI Service.
Loads and validates configuration from YAML file.
"""

import os
from typing import Any, Dict, List, Optional

import yaml


class ConfigLoader:
    """Configuration loader and validator."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self._config: Optional[Dict[str, Any]] = None

    def load(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file not found
            yaml.YAMLError: If YAML parsing fails
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

        if self._config is None:
            raise ValueError("Configuration file is empty")

        self._validate()
        return self._config

    def _validate(self) -> None:
        """
        Validate configuration structure.

        Raises:
            ValueError: If configuration is invalid
        """
        if self._config is None:
            raise ValueError("Configuration not loaded")

        # Validate required sections
        required_sections = ["tcp", "decoder", "inference", "tracker", "logging"]
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate TCP configuration
        tcp_config = self._config["tcp"]
        if "host" not in tcp_config or "port" not in tcp_config:
            raise ValueError("TCP configuration must include 'host' and 'port'")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation).

        Args:
            key: Configuration key (e.g., "tcp.port")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        if self._config is None:
            raise ValueError("Configuration not loaded. Call load() first.")

        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_tcp_config(self) -> Dict[str, Any]:
        """Get TCP server configuration."""
        return self.get("tcp", {})

    def get_decoder_config(self) -> Dict[str, Any]:
        """Get decoder configuration."""
        return self.get("decoder", {})

    def get_tracker_config(self) -> Dict[str, Any]:
        """Get tracker configuration."""
        return self.get("tracker", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get("logging", {})


# Global config instance
_config_instance: Optional[ConfigLoader] = None


def load_config(config_path: str = "config/config.yaml") -> ConfigLoader:
    """
    Load global configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        ConfigLoader instance
    """
    global _config_instance
    _config_instance = ConfigLoader(config_path)
    _config_instance.load()
    return _config_instance


def get_config() -> ConfigLoader:
    """
    Get global configuration instance.

    Returns:
        ConfigLoader instance

    Raises:
        RuntimeError: If configuration not loaded
    """
    if _config_instance is None:
        raise RuntimeError("Configuration not loaded. Call load_config() first.")
    return _config_instance
