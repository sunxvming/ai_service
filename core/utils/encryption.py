"""
Password encryption utilities using Fernet symmetric encryption.
"""

import os
import base64
from typing import Optional
from cryptography.fernet import Fernet


class PasswordEncryption:
    """Handles encryption and decryption of camera passwords."""

    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize encryption with a key.

        Args:
            key: Fernet key (32 url-safe base64-encoded bytes).
                 If None, loads from environment variable or generates new key.
        """
        if key is None:
            key = self._load_or_generate_key()

        self.cipher = Fernet(key)

    def encrypt(self, password: str) -> str:
        """
        Encrypt password.

        Args:
            password: Plain text password

        Returns:
            Encrypted password (base64 string)
        """
        if not password:
            return ""

        encrypted_bytes = self.cipher.encrypt(password.encode('utf-8'))
        return encrypted_bytes.decode('utf-8')

    def decrypt(self, encrypted_password: str) -> str:
        """
        Decrypt password.

        Args:
            encrypted_password: Encrypted password (base64 string)

        Returns:
            Plain text password
        """
        if not encrypted_password:
            return ""

        try:
            decrypted_bytes = self.cipher.decrypt(encrypted_password.encode('utf-8'))
            return decrypted_bytes.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to decrypt password: {e}")

    def _load_or_generate_key(self) -> bytes:
        """
        Load encryption key from environment or generate a new one.

        Returns:
            Fernet key
        """
        key_env = os.getenv('CAMERA_ENCRYPTION_KEY')

        if key_env:
            # Load from environment
            try:
                return base64.urlsafe_b64decode(key_env)
            except Exception as e:
                raise ValueError(f"Invalid CAMERA_ENCRYPTION_KEY format: {e}")
        else:
            # Generate new key
            key = Fernet.generate_key()
            key_str = base64.urlsafe_b64encode(key).decode('utf-8')

            # Save to .env file for persistence
            env_file = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
            try:
                with open(env_file, 'a') as f:
                    f.write(f'\nCAMERA_ENCRYPTION_KEY={key_str}\n')
                print(f"Generated new encryption key and saved to {env_file}")
            except Exception as e:
                print(f"Warning: Could not save encryption key to .env: {e}")
                print(f"Please set environment variable: CAMERA_ENCRYPTION_KEY={key_str}")

            return key


# Global instance
_encryptor: Optional[PasswordEncryption] = None


def get_encryptor() -> PasswordEncryption:
    """Get global PasswordEncryption instance (singleton)."""
    global _encryptor
    if _encryptor is None:
        _encryptor = PasswordEncryption()
    return _encryptor
