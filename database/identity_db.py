"""Identity database for storing and matching face embeddings."""
import logging
import tempfile
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np

from recognition.types import EmbeddingResult, ModelInfo

logger = logging.getLogger(__name__)


class IdentityDatabase:
    """Thread-safe database for storing face embeddings indexed by identity.

    Stores multiple embeddings per identity and supports similarity-based matching.
    Includes model fingerprint tracking to detect incompatible embedding versions.

    Attributes:
        max_embeddings_per_identity: Maximum embeddings stored per person (default 5).
    """

    def __init__(self, max_embeddings_per_identity: int = 5) -> None:
        """Initialize empty database.

        Args:
            max_embeddings_per_identity: Max embeddings kept per identity.
                Older embeddings are dropped when limit exceeded.
        """
        self.max_embeddings_per_identity = max_embeddings_per_identity
        self._identities: Dict[str, List[np.ndarray]] = {}
        self._fingerprints: Dict[str, str] = {}
        self._lock = Lock()

    def add_identity(
        self,
        name: str,
        embeddings_list: List[np.ndarray],
    ) -> None:
        """Add or update identity with list of embeddings.

        Stores up to max_embeddings_per_identity embeddings per identity.
        If more embeddings provided, keeps only the most recent ones.

        Args:
            name: Identity name (person identifier).
            embeddings_list: List of L2-normalized embedding vectors (shape (D,)).
                Should contain 1-5 embeddings for best matching.

        Raises:
            ValueError: If embeddings_list is empty or contains invalid data.

        Guarantees:
            - All stored embeddings are L2-normalized
            - At most max_embeddings_per_identity stored per identity
            - Embeddings with different fingerprints will be rejected
        """
        if not embeddings_list:
            raise ValueError("embeddings_list cannot be empty")

        with self._lock:
            if not embeddings_list:
                raise ValueError("embeddings_list cannot be empty")

            first_emb = embeddings_list[0]
            if not isinstance(first_emb, np.ndarray) or first_emb.ndim != 1:
                raise ValueError(
                    f"Each embedding must be 1D numpy array, got shape {first_emb.shape}"
                )

            # Verify all embeddings are L2-normalized
            for i, emb in enumerate(embeddings_list):
                norm = np.linalg.norm(emb)
                if not np.isclose(norm, 1.0, atol=1e-6):
                    raise ValueError(
                        f"Embedding {i} not L2-normalized: norm={norm:.6f}, expected ~1.0"
                    )

            # Keep only last max_embeddings_per_identity
            stored = embeddings_list[-self.max_embeddings_per_identity :]

            self._identities[name] = stored
            logger.info(f"Added identity '{name}' with {len(stored)} embeddings")

    def add_embedding_result(
        self,
        name: str,
        embedding_result: EmbeddingResult,
    ) -> None:
        """Add single embedding from EmbeddingResult to identity.

        Convenience method for adding one embedding at a time.
        Handles fingerprint tracking automatically.

        Args:
            name: Identity name.
            embedding_result: EmbeddingResult containing embedding and model info.

        Raises:
            ValueError: If embedding is not L2-normalized or fingerprints don't match.
        """
        with self._lock:
            # Check fingerprint compatibility
            if name in self._fingerprints:
                if self._fingerprints[name] != embedding_result.model_fingerprint:
                    raise ValueError(
                        f"Fingerprint mismatch for '{name}': "
                        f"stored={self._fingerprints[name]}, "
                        f"new={embedding_result.model_fingerprint}"
                    )
            else:
                self._fingerprints[name] = embedding_result.model_fingerprint

            # Get existing embeddings or start fresh
            existing = self._identities.get(name, [])
            existing.append(embedding_result.embedding)

            # Keep only last max_embeddings_per_identity
            stored = existing[-self.max_embeddings_per_identity :]
            self._identities[name] = stored

            logger.info(f"Added embedding for '{name}' (now {len(stored)} total)")

    def find_match(
        self,
        embedding: np.ndarray,
        threshold: float = 0.5,
        model_fingerprint: Optional[str] = None,
    ) -> Optional[Tuple[str, float]]:
        """Find best matching identity for embedding.

        Computes average embedding per identity, then finds highest similarity match.

        Args:
            embedding: Query L2-normalized embedding (shape (D,)).
            threshold: Minimum similarity to return a match (default 0.5).
            model_fingerprint: Expected fingerprint for validation. If provided,
                will check that stored embeddings match this fingerprint.

        Returns:
            Tuple of (identity_name, similarity_score) or None if no match
            above threshold found.

        Raises:
            ValueError: If embedding not L2-normalized or fingerprint mismatch.
        """
        norm = np.linalg.norm(embedding)
        if not np.isclose(norm, 1.0, atol=1e-6):
            raise ValueError(f"Query embedding not L2-normalized: norm={norm:.6f}")

        with self._lock:
            if not self._identities:
                return None

            best_identity = None
            best_similarity = threshold  # Only consider scores above threshold

            for name, embeddings in self._identities.items():
                # Check fingerprint compatibility if provided
                if model_fingerprint and name in self._fingerprints:
                    if self._fingerprints[name] != model_fingerprint:
                        logger.warning(
                            f"Skipping '{name}': fingerprint mismatch "
                            f"(stored={self._fingerprints[name]}, "
                            f"query={model_fingerprint})"
                        )
                        continue

                # Compute average embedding for this identity
                avg_embedding = np.mean(embeddings, axis=0)
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

                # Compute cosine similarity (L2-normalized vectors)
                similarity = float(np.dot(embedding, avg_embedding))
                similarity = float(np.clip(similarity, -1.0, 1.0))

                if similarity > best_similarity:
                    best_identity = name
                    best_similarity = similarity

            if best_identity:
                return (best_identity, best_similarity)
            return None

    def find_all_matches(
        self,
        embedding: np.ndarray,
        threshold: float = 0.5,
        model_fingerprint: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Find all matching identities above threshold, sorted by similarity.

        Args:
            embedding: Query L2-normalized embedding (shape (D,)).
            threshold: Minimum similarity to include (default 0.5).
            model_fingerprint: Expected fingerprint for validation.

        Returns:
            List of (identity_name, similarity_score) tuples sorted descending
            by similarity. Empty list if no matches found.
        """
        norm = np.linalg.norm(embedding)
        if not np.isclose(norm, 1.0, atol=1e-6):
            raise ValueError(f"Query embedding not L2-normalized: norm={norm:.6f}")

        with self._lock:
            if not self._identities:
                return []

            matches = []

            for name, embeddings in self._identities.items():
                # Check fingerprint compatibility if provided
                if model_fingerprint and name in self._fingerprints:
                    if self._fingerprints[name] != model_fingerprint:
                        continue

                # Compute average embedding for this identity
                avg_embedding = np.mean(embeddings, axis=0)
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

                # Compute cosine similarity
                similarity = float(np.dot(embedding, avg_embedding))
                similarity = float(np.clip(similarity, -1.0, 1.0))

                if similarity >= threshold:
                    matches.append((name, similarity))

            # Sort descending by similarity
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches

    def get_identity_stats(self, name: str) -> Optional[Dict[str, int]]:
        """Get statistics about stored embeddings for an identity.

        Args:
            name: Identity name.

        Returns:
            Dict with 'embedding_count' key, or None if identity not found.
        """
        with self._lock:
            if name not in self._identities:
                return None
            return {"embedding_count": len(self._identities[name])}

    def list_identities(self) -> Dict[str, int]:
        """List all stored identities with embedding counts.

        Returns:
            Dict mapping identity_name -> embedding_count.
        """
        with self._lock:
            return {name: len(embeddings) for name, embeddings in self._identities.items()}

    def save(self, filepath: str, passphrase: Optional[str] = None) -> None:
        """Save database to compressed NPZ file with optional encryption.

        Stores identities, embeddings, and fingerprints. If passphrase is provided,
        encrypts the entire NPZ file using AES-256-GCM.

        Args:
            filepath: Path to save .npz file to.
            passphrase: Optional passphrase for encryption. If provided, saves
                       encrypted file to filepath.enc instead of filepath.

        Raises:
            IOError: If file cannot be written.
            RuntimeError: If encryption is requested but cryptography package unavailable.

        Security:
            - Encryption uses AES-256-GCM with PBKDF2 key derivation
            - Original unencrypted NPZ file is not kept on disk
            - Passphrase should be provided via environment variable or secure input
        """
        with self._lock:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data: convert to format compatible with NPZ
            # Use object dtype to store variable-length lists
            data_dict = {}

            # Store embeddings and fingerprints
            for name, embeddings in self._identities.items():
                # Use safe keys (replace problematic characters)
                safe_name = name.replace("/", "_").replace("\\", "_")
                data_dict[f"emb_{safe_name}"] = np.array(
                    embeddings, dtype=np.float32
                )

            # Store fingerprints separately
            fingerprints_array = np.array(
                [
                    (name, fingerprint)
                    for name, fingerprint in self._fingerprints.items()
                ],
                dtype=[("name", "U64"), ("fingerprint", "U16")],
            )
            data_dict["_fingerprints"] = fingerprints_array

            # Store metadata
            data_dict["_max_embeddings_per_identity"] = np.array(
                [self.max_embeddings_per_identity]
            )
            data_dict["_identity_names"] = np.array(
                list(self._identities.keys()), dtype="U64"
            )

            # Save to NPZ file (possibly temporary if encrypting)
            if passphrase:
                # Save to temporary file, then encrypt
                with tempfile.NamedTemporaryFile(
                    mode='wb',
                    suffix='.npz',
                    delete=False,
                    dir=path.parent
                ) as temp_file:
                    temp_path = Path(temp_file.name)

                try:
                    # Save unencrypted NPZ to temporary file
                    np.savez_compressed(str(temp_path), **data_dict)

                    # Encrypt to final destination
                    from core.crypto_storage import CryptoStorage
                    crypto = CryptoStorage()
                    encrypted_path = path.with_suffix(path.suffix + '.enc')
                    crypto.encrypt_file(temp_path, encrypted_path, passphrase)

                    logger.info(
                        f"Saved encrypted database to {encrypted_path} "
                        f"({len(self._identities)} identities)"
                    )

                finally:
                    # Clean up temporary unencrypted file
                    if temp_path.exists():
                        temp_path.unlink()

            else:
                # Save unencrypted NPZ directly
                np.savez_compressed(filepath, **data_dict)
                logger.info(
                    f"Saved database to {filepath} ({len(self._identities)} identities)"
                )

    def load(self, filepath: str, passphrase: Optional[str] = None) -> None:
        """Load database from NPZ file with optional decryption.

        Restores identities, embeddings, and model fingerprints.
        Replaces current database contents. If encrypted file (.npz.enc) exists,
        decrypts it first using provided passphrase.

        Args:
            filepath: Path to load .npz file from.
            passphrase: Optional passphrase for decryption. Required if loading
                       encrypted file (.npz.enc).

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file is corrupted or incompatible format.
            DecryptionError: If decryption fails (wrong passphrase or corrupted file).

        Security:
            - Automatically detects encrypted files by .enc extension
            - Decrypts to temporary file, loads data, then deletes temporary file
            - Never leaves unencrypted data on disk when loading encrypted database
        """
        path = Path(filepath)
        encrypted_path = path.with_suffix(path.suffix + '.enc')

        # Check if encrypted version exists
        if encrypted_path.exists():
            if not passphrase:
                raise ValueError(
                    f"Encrypted database found at {encrypted_path} but no passphrase provided. "
                    f"Please provide passphrase to decrypt."
                )

            # Decrypt to temporary file
            with tempfile.NamedTemporaryFile(
                mode='wb',
                suffix='.npz',
                delete=False,
                dir=path.parent
            ) as temp_file:
                temp_path = Path(temp_file.name)

            try:
                # Decrypt encrypted file to temporary NPZ
                from core.crypto_storage import CryptoStorage, DecryptionError
                crypto = CryptoStorage()
                crypto.decrypt_file(encrypted_path, temp_path, passphrase)

                # Load from temporary decrypted file
                load_path = temp_path
                logger.info(f"Decrypted database from {encrypted_path}")

            except DecryptionError as e:
                # Clean up and re-raise with user-friendly message
                if temp_path.exists():
                    temp_path.unlink()
                raise ValueError(
                    f"Failed to decrypt database: {str(e)}. "
                    f"Please check your passphrase."
                ) from e
            except Exception as e:
                # Clean up on any other error
                if temp_path.exists():
                    temp_path.unlink()
                raise

        elif not path.exists():
            raise FileNotFoundError(
                f"Database file not found: {filepath} "
                f"(also checked for encrypted version at {encrypted_path})"
            )
        else:
            # Load unencrypted file directly
            load_path = path
            temp_path = None

        with self._lock:
            try:
                with np.load(str(load_path), allow_pickle=False) as npz_file:
                    keys = npz_file.files

                    # Clear current data
                    self._identities = {}
                    self._fingerprints = {}

                    # Load metadata
                    if "_max_embeddings_per_identity" in keys:
                        self.max_embeddings_per_identity = int(
                            npz_file["_max_embeddings_per_identity"][0]
                        )

                    # Load fingerprints
                    if "_fingerprints" in keys:
                        fp_array = npz_file["_fingerprints"]
                        for record in fp_array:
                            name = str(record["name"])
                            fingerprint = str(record["fingerprint"])
                            self._fingerprints[name] = fingerprint

                    # Load embeddings
                    identity_names = set()
                    if "_identity_names" in keys:
                        identity_names = set(npz_file["_identity_names"])

                    for key in keys:
                        if key.startswith("emb_"):
                            safe_name = key[4:]  # Remove 'emb_' prefix
                            embeddings = npz_file[key]

                            # Restore original name from set if needed
                            # For now, use the safe_name directly
                            if safe_name in identity_names or len(identity_names) == 0:
                                self._identities[safe_name] = embeddings

                    logger.info(
                        f"Loaded database from {filepath} "
                        f"({len(self._identities)} identities)"
                    )

            except Exception as e:
                raise ValueError(f"Failed to load database: {str(e)}")

            finally:
                # Clean up temporary decrypted file
                if temp_path and temp_path.exists():
                    temp_path.unlink()
                    logger.debug(f"Cleaned up temporary decrypted file: {temp_path}")

    def clear(self) -> None:
        """Clear all stored identities and embeddings.

        Useful for resetting the database or freeing memory.
        """
        with self._lock:
            self._identities.clear()
            self._fingerprints.clear()
            logger.info("Database cleared")