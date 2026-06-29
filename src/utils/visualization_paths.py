"""Shared path and metadata helpers for structure visualization scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path

PDB_PATTERN = "*_unrelaxed_*_alphafold2_model_*_seed_*.pdb"


def pdb_sort_key(path: Path) -> tuple[int, int, int, str]:
    """Sort AlphaFold predictions by rank, then model, then seed."""
    match = re.search(r"rank_(\d+).*model_(\d+)_seed_(\d+)", path.name)
    if match is None:
        return (999, 999, 999, path.name)
    rank, model, seed = (int(part) for part in match.groups())
    return (rank, model, seed, path.name)


def load_metadata(base_repo_path: Path) -> dict:
    with open(base_repo_path / "data" / "metadata.json") as f:
        return json.load(f)


def reference_pdb_for_state(
    base_repo_path: Path,
    metadata: dict,
    protein: str,
    state_key: str,
) -> Path:
    state = metadata[protein]["conformations"][state_key]
    name = f"{state['label']}_{state['pdb_id']}_{state['chain']}.pdb"
    return base_repo_path / "data" / protein / "references" / name


def reference_slug(metadata: dict, protein: str, state_key: str) -> str:
    state = metadata[protein]["conformations"][state_key]
    return f"{state['label']}_{state['pdb_id']}_{state['chain']}"


def original_fasta_path(base_repo_path: Path, protein: str) -> Path:
    return base_repo_path / "data" / protein / f"{protein}.fasta"
