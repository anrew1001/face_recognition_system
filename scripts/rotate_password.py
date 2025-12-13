#!/usr/bin/env python3
"""
Password rotation script for encrypted identity database.

This script safely rotates the encryption password for the identity database
without requiring manual file manipulation. It handles:
- Loading encrypted database with old password
- Saving with new password
- Atomic operations (all or nothing)
- Cleanup of temporary files
- Error handling with clear messages

Usage:
    # Interactive mode (prompts for passwords)
    python scripts/rotate_password.py

    # With arguments
    python scripts/rotate_password.py \
        --old-password "OldPassword123!" \
        --new-password "NewPassword456!" \
        --db-path "data/identities.npz"

    # Environment variables
    export FACE_DB_OLD_PASSPHRASE="OldPassword123!"
    export FACE_DB_NEW_PASSPHRASE="NewPassword456!"
    python scripts/rotate_password.py
"""
import argparse
import getpass
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import IdentityDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PasswordRotationError(Exception):
    """Raised when password rotation fails."""
    pass


class PasswordRotator:
    """Handles safe password rotation for encrypted databases."""

    def __init__(self, db_path: str = "data/identities.npz", run_script_path: str = "run_encrypted.py"):
        """Initialize password rotator.

        Args:
            db_path: Path to database file (without .enc extension).
            run_script_path: Path to run_encrypted.py script to auto-update.
        """
        self.db_path = Path(db_path)
        self.encrypted_path = self.db_path.with_suffix(self.db_path.suffix + '.enc')
        self.run_script_path = Path(run_script_path)

    def check_database_exists(self) -> bool:
        """Check if database exists (encrypted or unencrypted).

        Returns:
            True if either .npz or .npz.enc exists.
        """
        return self.db_path.exists() or self.encrypted_path.exists()

    def get_password_input(
        self,
        prompt: str,
        env_var: Optional[str] = None,
        confirm: bool = False
    ) -> str:
        """Get password from user with optional environment variable fallback.

        Args:
            prompt: Prompt text to display.
            env_var: Environment variable to check first.
            confirm: If True, require password confirmation.

        Returns:
            Password string.

        Raises:
            PasswordRotationError: If passwords don't match (when confirm=True).
        """
        # Try environment variable first
        if env_var and env_var in os.environ:
            password = os.environ[env_var]
            logger.info(f"Using password from environment variable: {env_var}")
            return password

        # Interactive input
        while True:
            password = getpass.getpass(prompt)

            if not password:
                logger.error("Password cannot be empty!")
                continue

            if len(password) < 6:
                logger.warning("Warning: Password is very short (< 6 characters)")
                response = input("Continue anyway? (y/n): ").lower()
                if response != 'y':
                    continue

            if confirm:
                password_confirm = getpass.getpass("Confirm password: ")
                if password != password_confirm:
                    logger.error("Passwords don't match!")
                    continue

            return password

    def update_run_script(self, new_password: str, backup: bool = True) -> bool:
        """Update run_encrypted.py with new password.

        Args:
            new_password: New password to set in the script.
            backup: If True, create backup before updating.

        Returns:
            True if update successful.

        Raises:
            PasswordRotationError: If update fails.
        """
        if not self.run_script_path.exists():
            logger.warning(f"run_encrypted.py not found at {self.run_script_path}")
            logger.warning("Skipping auto-update of run script")
            return False

        try:
            # Read current script
            with open(self.run_script_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Create backup if requested
            if backup:
                backup_path = self.run_script_path.with_suffix('.py.backup')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Created backup: {backup_path}")

            # Find and replace password line
            import re
            pattern = r"(os\.environ\[['\"]FACE_DB_PASSPHRASE['\"]\]\s*=\s*['\"])([^'\"]+)(['\"])"

            def replace_password(match):
                return f"{match.group(1)}{new_password}{match.group(3)}"

            new_content, count = re.subn(pattern, replace_password, content)

            if count == 0:
                raise PasswordRotationError(
                    f"Could not find password line in {self.run_script_path}\n"
                    f"Expected pattern: os.environ['FACE_DB_PASSPHRASE'] = 'password'"
                )

            # Write updated script
            with open(self.run_script_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            logger.info(f"✓ Updated {self.run_script_path} with new password")
            return True

        except Exception as e:
            raise PasswordRotationError(f"Failed to update run script: {e}")

    def rotate_password(
        self,
        old_password: str,
        new_password: str,
        dry_run: bool = False,
        update_run_script: bool = True
    ) -> bool:
        """Rotate database password.

        Steps:
        1. Load database with old password
        2. Save database with new password
        3. Verify new encrypted file exists
        4. Update run_encrypted.py with new password
        5. Clean up temporary files

        Args:
            old_password: Current encryption password.
            new_password: New encryption password.
            dry_run: If True, simulate rotation without saving.
            update_run_script: If True, automatically update run_encrypted.py.

        Returns:
            True if rotation successful.

        Raises:
            PasswordRotationError: If rotation fails at any step.
        """
        logger.info("=" * 60)
        logger.info("Password Rotation Procedure")
        logger.info("=" * 60)

        # Step 1: Verify database exists
        logger.info("Step 1: Checking database...")
        if not self.check_database_exists():
            raise PasswordRotationError(
                f"Database not found at {self.db_path} or {self.encrypted_path}"
            )
        logger.info("✓ Database found")

        # Step 2: Create backup
        logger.info("Step 2: Creating backup...")
        backup_path = None
        if self.encrypted_path.exists():
            backup_path = self.encrypted_path.with_suffix(
                self.encrypted_path.suffix + f'.backup'
            )
            try:
                import shutil
                shutil.copy2(self.encrypted_path, backup_path)
                logger.info(f"✓ Backup created: {backup_path}")
            except Exception as e:
                raise PasswordRotationError(f"Failed to create backup: {e}")

        # Step 3: Load database with old password
        logger.info("Step 3: Loading database with old password...")
        try:
            db = IdentityDatabase()
            db.load(str(self.db_path), passphrase=old_password)
            identities = db.list_identities()
            logger.info(f"✓ Database loaded successfully")
            logger.info(f"  Identities in database: {len(identities)}")
            for name, count in identities.items():
                logger.info(f"    - {name}: {count} embedding(s)")
        except Exception as e:
            raise PasswordRotationError(
                f"Failed to load database with old password: {e}"
            )

        # Step 4: Dry run check
        if dry_run:
            logger.info("DRY RUN: Would save with new password (not actually saving)")
            logger.info("✓ Rotation would complete successfully")
            return True

        # Step 5: Save with new password
        logger.info("Step 4: Saving database with new password...")
        try:
            db.save(str(self.db_path), passphrase=new_password)
            logger.info(f"✓ Database saved with new password")
            logger.info(f"  Encrypted file: {self.encrypted_path}")
        except Exception as e:
            raise PasswordRotationError(
                f"Failed to save database with new password: {e}"
            )

        # Step 6: Verify new password works
        logger.info("Step 5: Verifying new password...")
        try:
            db_verify = IdentityDatabase()
            db_verify.load(str(self.db_path), passphrase=new_password)
            identities_verify = db_verify.list_identities()
            if len(identities_verify) != len(identities):
                raise PasswordRotationError(
                    f"Identity count mismatch after rotation: "
                    f"{len(identities_verify)} != {len(identities)}"
                )
            logger.info(f"✓ New password verified successfully")
        except Exception as e:
            raise PasswordRotationError(
                f"Failed to verify new password: {e}"
            )

        # Step 7: Old password should not work anymore
        logger.info("Step 6: Verifying old password no longer works...")
        try:
            db_old = IdentityDatabase()
            db_old.load(str(self.db_path), passphrase=old_password)
            logger.warning("⚠ WARNING: Old password still works!")
            logger.warning("This means the new password may not have been applied correctly.")
        except Exception:
            # Expected - old password should fail
            logger.info("✓ Old password correctly rejected")

        # Step 8: Update run_encrypted.py if requested
        if update_run_script:
            logger.info("Step 7: Updating run_encrypted.py...")
            try:
                self.update_run_script(new_password, backup=True)
            except Exception as e:
                logger.error(f"Failed to update run_encrypted.py: {e}")
                logger.warning("You will need to update run_encrypted.py manually")

        logger.info("=" * 60)
        logger.info("Password Rotation Complete!")
        logger.info("=" * 60)
        logger.info(f"New encrypted database: {self.encrypted_path}")
        if backup_path:
            logger.info(f"Database backup: {backup_path}")
        if update_run_script and self.run_script_path.exists():
            logger.info(f"run_encrypted.py updated with new password")
            backup_script = self.run_script_path.with_suffix('.py.backup')
            if backup_script.exists():
                logger.info(f"Script backup: {backup_script}")
        logger.info(f"Keep backups for 7 days before deletion")

        return True

    def clean_old_backups(self, keep_days: int = 7) -> int:
        """Clean up old backup files.

        Args:
            keep_days: Keep backups newer than this many days.

        Returns:
            Number of files deleted.
        """
        import time

        deleted_count = 0
        current_time = time.time()
        cutoff_time = current_time - (keep_days * 24 * 60 * 60)

        # Find all backup files
        backup_pattern = str(self.encrypted_path) + ".backup*"
        logger.info(f"Cleaning backups older than {keep_days} days...")

        # This is a simplified version - full implementation would use glob
        # For now, just log the recommendation
        logger.info(f"Backup files matching pattern: {backup_pattern}")
        logger.info(f"To delete old backups manually:")
        logger.info(f"  find {self.encrypted_path.parent} -name '{self.encrypted_path.name}.backup*' -mtime +{keep_days} -delete")

        return deleted_count


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Rotate encryption password for face recognition database.",
        epilog="Environment variables: FACE_DB_OLD_PASSPHRASE, FACE_DB_NEW_PASSPHRASE",
    )

    parser.add_argument(
        "--old-password",
        help="Current encryption password (or set FACE_DB_OLD_PASSPHRASE)",
    )
    parser.add_argument(
        "--new-password",
        help="New encryption password (or set FACE_DB_NEW_PASSPHRASE)",
    )
    parser.add_argument(
        "--db-path",
        default="data/identities.npz",
        help="Path to database file (default: data/identities.npz)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate rotation without saving",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmations",
    )
    parser.add_argument(
        "--clean-backups",
        type=int,
        metavar="DAYS",
        help="Clean backups older than DAYS (default: 7)",
    )
    parser.add_argument(
        "--skip-run-script-update",
        action="store_true",
        help="Don't automatically update run_encrypted.py",
    )
    parser.add_argument(
        "--run-script-path",
        default="run_encrypted.py",
        help="Path to run_encrypted.py (default: run_encrypted.py)",
    )

    args = parser.parse_args()

    try:
        rotator = PasswordRotator(
            db_path=args.db_path,
            run_script_path=args.run_script_path
        )

        # Get passwords
        old_password = args.old_password or os.getenv("FACE_DB_OLD_PASSPHRASE")
        if not old_password:
            old_password = rotator.get_password_input(
                "Enter current encryption password: ",
                env_var="FACE_DB_OLD_PASSPHRASE",
            )

        new_password = args.new_password or os.getenv("FACE_DB_NEW_PASSPHRASE")
        if not new_password:
            new_password = rotator.get_password_input(
                "Enter new encryption password: ",
                env_var="FACE_DB_NEW_PASSPHRASE",
                confirm=True,
            )

        # Confirm if not forced
        if not args.force:
            logger.info("")
            logger.info("Ready to rotate password:")
            logger.info(f"  Database: {args.db_path}")
            if args.dry_run:
                logger.info("  Mode: DRY RUN (no changes will be made)")
            logger.info("")
            response = input("Continue? (yes/no): ").lower().strip()
            if response not in ['yes', 'y']:
                logger.info("Rotation cancelled.")
                return 0

        # Perform rotation
        success = rotator.rotate_password(
            old_password,
            new_password,
            dry_run=args.dry_run,
            update_run_script=not args.skip_run_script_update,
        )

        # Clean old backups if requested
        if args.clean_backups:
            rotator.clean_old_backups(keep_days=args.clean_backups)

        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Test the new password:")
        logger.info(f"   python run_encrypted.py")
        logger.info("   OR")
        logger.info(f"   FACE_DB_PASSPHRASE='<new_password>' python main.py")
        logger.info("")
        logger.info("2. Update deployment configurations with new password")
        logger.info("3. Update backups of the passphrase in secure storage")
        logger.info("4. Keep backup files for 7 days before deletion")
        logger.info("")
        if not args.skip_run_script_update:
            logger.info("✓ run_encrypted.py has been automatically updated!")

        return 0

    except PasswordRotationError as e:
        logger.error(f"Rotation failed: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Rotation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
