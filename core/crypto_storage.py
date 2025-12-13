"""AES-256-GCM encryption for biometric data storage.

This module provides file encryption/decryption using AES-256-GCM with PBKDF2
key derivation for secure storage of sensitive biometric embeddings.

Security Notes:
    - This is demonstration-level security for educational purposes
    - Production deployments should use hardware security modules (HSM)
    - Never store passphrases in code or version control
    - Use environment variables or secure key management systems
"""
import logging
import os
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


class CryptoStorageError(Exception):
    """Base exception for crypto storage operations."""
    pass


class DecryptionError(CryptoStorageError):
    """Raised when decryption fails (wrong passphrase or corrupted data)."""
    pass


class EncryptionError(CryptoStorageError):
    """Raised when encryption fails."""
    pass


class CryptoStorage:
    """Secure file encryption/decryption using AES-256-GCM.

    Uses PBKDF2 for key derivation from passphrase with random salt.
    Stores salt and nonce alongside encrypted data for proper decryption.

    File format (encrypted):
        [salt: 16 bytes][nonce: 12 bytes][tag: 16 bytes][ciphertext: variable]

    Example:
        >>> crypto = CryptoStorage()
        >>> crypto.encrypt_file("data.npz", "data.npz.enc", "my_passphrase")
        >>> crypto.decrypt_file("data.npz.enc", "data.npz", "my_passphrase")
    """

    # AES-256-GCM parameters
    KEY_SIZE = 32  # 256 bits
    SALT_SIZE = 16  # 128 bits
    NONCE_SIZE = 12  # 96 bits (recommended for GCM)
    TAG_SIZE = 16  # 128 bits

    # PBKDF2 parameters
    PBKDF2_ITERATIONS = 100000  # OWASP recommendation (2023)
    PBKDF2_ALGORITHM = "sha256"

    def __init__(self):
        """Initialize crypto storage.

        Raises:
            RuntimeError: If cryptography package is not installed.
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.backends import default_backend

            self.AESGCM = AESGCM
            self.PBKDF2HMAC = PBKDF2HMAC
            self.hashes = hashes
            self.backend = default_backend()
        except ImportError as e:
            raise RuntimeError(
                "cryptography package not installed. Install with: "
                "pip install cryptography>=41.0.0"
            ) from e

    def _derive_key(self, passphrase: str, salt: bytes) -> bytes:
        """Derive 256-bit encryption key from passphrase using PBKDF2.

        Args:
            passphrase: User-provided passphrase (UTF-8 string).
            salt: Random salt for key derivation (16 bytes).

        Returns:
            Derived 32-byte encryption key.
        """
        kdf = self.PBKDF2HMAC(
            algorithm=self.hashes.SHA256(),
            length=self.KEY_SIZE,
            salt=salt,
            iterations=self.PBKDF2_ITERATIONS,
            backend=self.backend
        )
        key = kdf.derive(passphrase.encode('utf-8'))
        return key

    def encrypt_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        passphrase: str
    ) -> None:
        """Encrypt file using AES-256-GCM.

        Generates random salt and nonce, derives key from passphrase,
        encrypts file contents, and writes encrypted file with metadata.

        Args:
            input_path: Path to plaintext file to encrypt.
            output_path: Path to write encrypted file.
            passphrase: Encryption passphrase (UTF-8 string).

        Raises:
            FileNotFoundError: If input file does not exist.
            EncryptionError: If encryption fails.

        Security:
            - Salt and nonce are randomly generated for each encryption
            - Never reuses nonce for same key (critical for GCM security)
            - Includes authentication tag to detect tampering
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        try:
            # Read plaintext data
            with open(input_path, 'rb') as f:
                plaintext = f.read()

            # Generate random salt and nonce
            salt = os.urandom(self.SALT_SIZE)
            nonce = os.urandom(self.NONCE_SIZE)

            # Derive key from passphrase
            key = self._derive_key(passphrase, salt)

            # Encrypt data with AES-256-GCM
            aesgcm = self.AESGCM(key)
            ciphertext = aesgcm.encrypt(nonce, plaintext, None)

            # Write encrypted file: [salt][nonce][ciphertext+tag]
            with open(output_path, 'wb') as f:
                f.write(salt)
                f.write(nonce)
                f.write(ciphertext)

            logger.info(
                f"Encrypted {input_path} -> {output_path} "
                f"({len(plaintext)} bytes plaintext, {len(ciphertext)} bytes ciphertext)"
            )

        except Exception as e:
            # Clean up partial output file on error
            if output_path.exists():
                output_path.unlink()
            raise EncryptionError(f"Encryption failed: {e}") from e

    def decrypt_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        passphrase: str
    ) -> None:
        """Decrypt file encrypted with AES-256-GCM.

        Reads encrypted file, extracts salt and nonce, derives key,
        decrypts and authenticates ciphertext.

        Args:
            input_path: Path to encrypted file.
            output_path: Path to write decrypted plaintext.
            passphrase: Decryption passphrase (must match encryption passphrase).

        Raises:
            FileNotFoundError: If input file does not exist.
            DecryptionError: If decryption fails (wrong passphrase, corrupted data, or tampered file).

        Security:
            - Verifies authentication tag before returning data
            - Detects any modification to encrypted file
            - Wrong passphrase or corrupted data raises DecryptionError
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Encrypted file not found: {input_path}")

        try:
            # Read encrypted file
            with open(input_path, 'rb') as f:
                encrypted_data = f.read()

            # Validate file size
            min_size = self.SALT_SIZE + self.NONCE_SIZE + self.TAG_SIZE
            if len(encrypted_data) < min_size:
                raise DecryptionError(
                    f"File too small to be valid encrypted file "
                    f"(expected >= {min_size} bytes, got {len(encrypted_data)})"
                )

            # Extract salt, nonce, and ciphertext
            salt = encrypted_data[:self.SALT_SIZE]
            nonce = encrypted_data[self.SALT_SIZE:self.SALT_SIZE + self.NONCE_SIZE]
            ciphertext = encrypted_data[self.SALT_SIZE + self.NONCE_SIZE:]

            # Derive key from passphrase
            key = self._derive_key(passphrase, salt)

            # Decrypt and authenticate
            aesgcm = self.AESGCM(key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)

            # Write decrypted file
            with open(output_path, 'wb') as f:
                f.write(plaintext)

            logger.info(
                f"Decrypted {input_path} -> {output_path} "
                f"({len(ciphertext)} bytes ciphertext, {len(plaintext)} bytes plaintext)"
            )

        except Exception as e:
            # Clean up partial output file on error
            if output_path.exists():
                output_path.unlink()

            # Provide user-friendly error message
            if "MAC check failed" in str(e) or "authentication" in str(e).lower():
                raise DecryptionError(
                    "Decryption failed: Invalid passphrase or corrupted file"
                ) from e
            else:
                raise DecryptionError(f"Decryption failed: {e}") from e

    def encrypt_bytes(self, data: bytes, passphrase: str) -> bytes:
        """Encrypt raw bytes (for in-memory operations).

        Args:
            data: Plaintext bytes to encrypt.
            passphrase: Encryption passphrase.

        Returns:
            Encrypted bytes: [salt][nonce][ciphertext+tag]
        """
        salt = os.urandom(self.SALT_SIZE)
        nonce = os.urandom(self.NONCE_SIZE)
        key = self._derive_key(passphrase, salt)

        aesgcm = self.AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, data, None)

        return salt + nonce + ciphertext

    def decrypt_bytes(self, encrypted_data: bytes, passphrase: str) -> bytes:
        """Decrypt raw bytes (for in-memory operations).

        Args:
            encrypted_data: Encrypted bytes from encrypt_bytes().
            passphrase: Decryption passphrase.

        Returns:
            Decrypted plaintext bytes.

        Raises:
            DecryptionError: If decryption fails.
        """
        min_size = self.SALT_SIZE + self.NONCE_SIZE + self.TAG_SIZE
        if len(encrypted_data) < min_size:
            raise DecryptionError("Invalid encrypted data size")

        salt = encrypted_data[:self.SALT_SIZE]
        nonce = encrypted_data[self.SALT_SIZE:self.SALT_SIZE + self.NONCE_SIZE]
        ciphertext = encrypted_data[self.SALT_SIZE + self.NONCE_SIZE:]

        key = self._derive_key(passphrase, salt)

        try:
            aesgcm = self.AESGCM(key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
            return plaintext
        except Exception as e:
            if "MAC check failed" in str(e) or "authentication" in str(e).lower():
                raise DecryptionError("Invalid passphrase or corrupted data") from e
            else:
                raise DecryptionError(f"Decryption failed: {e}") from e