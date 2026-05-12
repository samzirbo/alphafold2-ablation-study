"""Google Drive utilities for Colab environments."""

from __future__ import annotations

from pathlib import Path

DEFAULT_DRIVE_PATH = Path("/content/drive/MyDrive/Masters/Semester 3/AlphaFold2 Ablation Study/04_Results")


def get_drive_result_dir(job: str, drive_path: str | None = None) -> Path:
    """Return the result directory on Google Drive for a given job."""
    base = Path(drive_path) if drive_path else DEFAULT_DRIVE_PATH
    return base / job
