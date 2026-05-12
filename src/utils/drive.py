"""Google Drive authentication and file operations for Colab environments."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DRIVE_PATH = Path("/content/drive/MyDrive/Masters/Semester 3/AlphaFold2 Ablation Study/04_Results")


def mount_drive() -> None:
    """Mount Google Drive in a Colab environment."""
    from google.colab import drive
    drive.mount("/content/drive")
    logger.info("Google Drive mounted.")


def get_drive_result_dir(job: str, drive_path: str | None = None) -> Path:
    """Return the result directory on Google Drive for a given job."""
    base = Path(drive_path) if drive_path else DEFAULT_DRIVE_PATH
    return base / job
