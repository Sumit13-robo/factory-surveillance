"""
Privacy Utilities — Face blurring and log encryption.

Provides privacy-preserving features for GDPR/compliance:
  - Gaussian face blurring on detected person regions
  - Symmetric encryption (Fernet) for log files
"""

import os
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def blur_faces(
    frame: np.ndarray,
    detections: list,
    person_class: str = "person",
    blur_strength: int = 51,
) -> np.ndarray:
    """
    Apply Gaussian blur to the head region of detected persons.

    Estimates the head region as the top 30% of the person bounding box
    and applies a strong Gaussian blur.

    Args:
        frame: Input BGR frame.
        detections: List of detection dicts with 'bbox' and 'class_name'.
        person_class: Class name for person detections.
        blur_strength: Gaussian kernel size (must be odd).

    Returns:
        Frame with faces blurred.
    """
    if blur_strength % 2 == 0:
        blur_strength += 1

    result = frame.copy()

    for det in detections:
        if det.get("class_name") != person_class:
            continue

        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        h = y2 - y1

        # Estimate head region: top 30% of bounding box
        head_y2 = y1 + int(h * 0.3)
        head_region = result[y1:head_y2, x1:x2]

        if head_region.size == 0:
            continue

        # Apply strong Gaussian blur
        blurred = cv2.GaussianBlur(head_region, (blur_strength, blur_strength), 30)
        result[y1:head_y2, x1:x2] = blurred

    return result


def generate_encryption_key(key_file: str) -> bytes:
    """
    Generate and save a new Fernet encryption key.

    Args:
        key_file: Path to save the key file.

    Returns:
        Generated key bytes.
    """
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        logger.error("cryptography package not installed. Run: pip install cryptography")
        raise

    key = Fernet.generate_key()
    os.makedirs(os.path.dirname(key_file), exist_ok=True)
    with open(key_file, "wb") as f:
        f.write(key)
    os.chmod(key_file, 0o600)  # Restrict permissions
    logger.info(f"🔐 Encryption key generated: {key_file}")
    return key


def load_encryption_key(key_file: str) -> bytes:
    """
    Load encryption key from file, generating if it doesn't exist.

    Args:
        key_file: Path to the key file.

    Returns:
        Key bytes.
    """
    if not os.path.exists(key_file):
        return generate_encryption_key(key_file)
    with open(key_file, "rb") as f:
        return f.read()


def encrypt_file(filepath: str, key_file: str):
    """
    Encrypt a file in-place using Fernet symmetric encryption.

    Args:
        filepath: Path to the file to encrypt.
        key_file: Path to the encryption key file.
    """
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        logger.error("cryptography package not installed.")
        return

    key = load_encryption_key(key_file)
    fernet = Fernet(key)

    with open(filepath, "rb") as f:
        data = f.read()

    encrypted = fernet.encrypt(data)

    with open(filepath + ".enc", "wb") as f:
        f.write(encrypted)

    os.remove(filepath)
    logger.info(f"🔐 Encrypted: {filepath} → {filepath}.enc")


def decrypt_file(encrypted_path: str, key_file: str, output_path: Optional[str] = None):
    """
    Decrypt a Fernet-encrypted file.

    Args:
        encrypted_path: Path to the .enc file.
        key_file: Path to the encryption key file.
        output_path: Where to write decrypted content. Defaults to original name.
    """
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        logger.error("cryptography package not installed.")
        return

    key = load_encryption_key(key_file)
    fernet = Fernet(key)

    with open(encrypted_path, "rb") as f:
        encrypted = f.read()

    decrypted = fernet.decrypt(encrypted)

    if output_path is None:
        output_path = encrypted_path.replace(".enc", "")

    with open(output_path, "wb") as f:
        f.write(decrypted)

    logger.info(f"🔓 Decrypted: {encrypted_path} → {output_path}")
