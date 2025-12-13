#!/usr/bin/env python3
"""
Setup encryption for face recognition database.

This script helps you:
1. Convert unencrypted database to encrypted
2. Clean up old database files after encryption
3. Set environment variable for daily use
4. Test encryption is working

Usage:
    # Interactive setup
    python scripts/setup_encryption.py

    # Encrypt existing database
    python scripts/setup_encryption.py --encrypt

    # Cleanup old files after encryption
    python scripts/setup_encryption.py --cleanup

    # Show current status
    python scripts/setup_encryption.py --status
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import IdentityDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


class EncryptionSetup:
    """Helper for setting up database encryption."""

    def __init__(self, db_path: str = "data/identities.npz"):
        self.db_path = Path(db_path)
        self.encrypted_path = self.db_path.with_suffix(self.db_path.suffix + '.enc')
        self.unencrypted_path = self.db_path
        self.backup_path = self.encrypted_path.with_suffix(
            self.encrypted_path.suffix + '.backup'
        )

    def show_status(self) -> None:
        """Show current encryption status."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("Database Encryption Status")
        logger.info("=" * 60)

        unenc_exists = self.unencrypted_path.exists()
        enc_exists = self.encrypted_path.exists()
        backup_exists = self.backup_path.exists()

        logger.info(f"Unencrypted database: {self.unencrypted_path}")
        logger.info(f"  Status: {'✓ EXISTS' if unenc_exists else '✗ Missing'}")
        if unenc_exists:
            size = self.unencrypted_path.stat().st_size / 1024
            logger.info(f"  Size: {size:.1f} KB")

        logger.info("")
        logger.info(f"Encrypted database: {self.encrypted_path}")
        logger.info(f"  Status: {'✓ EXISTS' if enc_exists else '✗ Missing'}")
        if enc_exists:
            size = self.encrypted_path.stat().st_size / 1024
            logger.info(f"  Size: {size:.1f} KB")

        logger.info("")
        logger.info(f"Backup: {self.backup_path}")
        logger.info(f"  Status: {'✓ EXISTS' if backup_exists else '✗ Missing'}")

        logger.info("")
        if enc_exists and unenc_exists:
            logger.info("Status: MIXED - Both encrypted and unencrypted exist")
            logger.info("Recommendation: Run --cleanup to remove unencrypted copy")
            logger.info("")
            logger.info("To use encrypted database:")
            logger.info("  export FACE_DB_PASSPHRASE='your_password'")
            logger.info("  python main.py")
        elif enc_exists and not unenc_exists:
            logger.info("Status: ENCRYPTED - Ready to use")
            logger.info("")
            logger.info("To run the system:")
            logger.info("  export FACE_DB_PASSPHRASE='your_password'")
            logger.info("  python main.py")
        elif unenc_exists and not enc_exists:
            logger.info("Status: UNENCRYPTED - No encryption active")
            logger.info("Recommendation: Run --encrypt to enable encryption")
            logger.info("")
            logger.info("To encrypt:")
            logger.info("  python scripts/setup_encryption.py --encrypt")
        else:
            logger.info("Status: NO DATABASE - Create one by running main.py")

        logger.info("")
        logger.info("Environment variable: FACE_DB_PASSPHRASE")
        passphrase = os.getenv("FACE_DB_PASSPHRASE")
        if passphrase:
            logger.info(f"  Status: ✓ SET (value hidden for security)")
        else:
            logger.info(f"  Status: ✗ NOT SET")

        logger.info("=" * 60)
        logger.info("")

    def encrypt_database(self, passphrase: str) -> bool:
        """Encrypt unencrypted database.

        Args:
            passphrase: Password for encryption.

        Returns:
            True if successful.
        """
        logger.info("")
        logger.info("=" * 60)
        logger.info("Encrypting Database")
        logger.info("=" * 60)

        if not self.unencrypted_path.exists():
            logger.error(f"Unencrypted database not found: {self.unencrypted_path}")
            return False

        try:
            logger.info("Step 1: Loading unencrypted database...")
            db = IdentityDatabase()
            db.load(str(self.unencrypted_path))
            identities = db.list_identities()
            logger.info(f"✓ Loaded {len(identities)} identities")
            for name, count in identities.items():
                logger.info(f"  - {name}: {count} embedding(s)")

            logger.info("")
            logger.info("Step 2: Saving with encryption...")
            db.save(str(self.db_path), passphrase=passphrase)
            logger.info(f"✓ Database encrypted: {self.encrypted_path}")

            logger.info("")
            logger.info("Step 3: Verifying encryption...")
            db_verify = IdentityDatabase()
            db_verify.load(str(self.db_path), passphrase=passphrase)
            identities_verify = db_verify.list_identities()
            if len(identities_verify) == len(identities):
                logger.info(f"✓ Verification successful: {len(identities_verify)} identities")
            else:
                logger.error("✗ Identity count mismatch!")
                return False

            logger.info("")
            logger.info("=" * 60)
            logger.info("✓ Encryption Complete!")
            logger.info("=" * 60)
            logger.info("")
            logger.info("Old file still exists: " + str(self.unencrypted_path))
            logger.info("You can safely delete it with:")
            logger.info(f"  rm {self.unencrypted_path}")
            logger.info("")
            logger.info("Or run:")
            logger.info("  python scripts/setup_encryption.py --cleanup")
            logger.info("")

            return True

        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return False

    def cleanup_old_files(self) -> bool:
        """Remove unencrypted database after encryption.

        Returns:
            True if successful.
        """
        logger.info("")
        logger.info("=" * 60)
        logger.info("Cleanup - Remove Old Unencrypted Files")
        logger.info("=" * 60)

        if not self.encrypted_path.exists():
            logger.error("Encrypted database not found - cannot cleanup safely!")
            logger.info("Make sure you have: " + str(self.encrypted_path))
            return False

        if not self.unencrypted_path.exists():
            logger.info("No unencrypted files to clean up")
            return True

        try:
            logger.info(f"Removing: {self.unencrypted_path}")
            self.unencrypted_path.unlink()
            logger.info("✓ Removed unencrypted database")

            logger.info("")
            logger.info("=" * 60)
            logger.info("✓ Cleanup Complete!")
            logger.info("=" * 60)
            logger.info("")
            logger.info("Your encrypted database is ready:")
            logger.info(f"  {self.encrypted_path}")
            logger.info("")
            logger.info("To run the system:")
            logger.info("  export FACE_DB_PASSPHRASE='your_password'")
            logger.info("  python main.py")
            logger.info("")

            return True

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False

    def interactive_setup(self) -> bool:
        """Interactive encryption setup wizard.

        Returns:
            True if successful.
        """
        logger.info("")
        logger.info("=" * 60)
        logger.info("Database Encryption Setup Wizard")
        logger.info("=" * 60)
        logger.info("")

        # Show current status
        self.show_status()

        # Check what needs to be done
        if not self.unencrypted_path.exists() and not self.encrypted_path.exists():
            logger.error("No database found!")
            logger.info("Please run the main application first to create a database:")
            logger.info("  python main.py")
            return False

        if self.encrypted_path.exists() and self.unencrypted_path.exists():
            logger.warning("Both encrypted and unencrypted files exist!")
            response = input("\nCleanup old unencrypted file? (yes/no): ").lower().strip()
            if response in ['yes', 'y']:
                return self.cleanup_old_files()
            return False

        if self.unencrypted_path.exists() and not self.encrypted_path.exists():
            logger.info("Ready to encrypt your database!")
            response = input("\nEncrypt now? (yes/no): ").lower().strip()
            if response not in ['yes', 'y']:
                logger.info("Encryption skipped")
                return False

            import getpass
            passphrase = getpass.getpass("Enter encryption password: ")
            passphrase_confirm = getpass.getpass("Confirm password: ")

            if passphrase != passphrase_confirm:
                logger.error("Passwords don't match!")
                return False

            success = self.encrypt_database(passphrase)
            if success:
                logger.info("Next steps:")
                logger.info("1. Delete old unencrypted file (or run --cleanup):")
                logger.info(f"   rm {self.unencrypted_path}")
                logger.info("2. Set password for daily use:")
                logger.info(f"   export FACE_DB_PASSPHRASE='{passphrase}'")
                logger.info("3. Run the main application:")
                logger.info("   python main.py")

            return success

        if self.encrypted_path.exists() and not self.unencrypted_path.exists():
            logger.info("✓ Database is already encrypted and clean!")
            return True

        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup encryption for face recognition database",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current encryption status",
    )
    parser.add_argument(
        "--encrypt",
        action="store_true",
        help="Encrypt unencrypted database",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove old unencrypted files after encryption",
    )
    parser.add_argument(
        "--passphrase",
        help="Encryption passphrase (for --encrypt)",
    )
    parser.add_argument(
        "--db-path",
        default="data/identities.npz",
        help="Path to database (default: data/identities.npz)",
    )

    args = parser.parse_args()

    setup = EncryptionSetup(db_path=args.db_path)

    try:
        if args.status:
            setup.show_status()
            return 0

        if args.encrypt:
            if not args.passphrase:
                import getpass
                args.passphrase = getpass.getpass("Enter encryption password: ")

            success = setup.encrypt_database(args.passphrase)
            return 0 if success else 1

        if args.cleanup:
            success = setup.cleanup_old_files()
            return 0 if success else 1

        # Default: interactive wizard
        success = setup.interactive_setup()
        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("\nSetup cancelled")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
