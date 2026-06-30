"""Per-residue local TM-score contributions using ``tmtools``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tmtools import tm_align
from tmtools.io import get_residue_data, get_structure


@dataclass(frozen=True)
class ResidueTmScore:
    chain_id: str
    resi: int
    score: float


@dataclass(frozen=True)
class PerResidueTmScoreResult:
    tm_score: float
    residues: tuple[ResidueTmScore, ...]


def tm_score_d0(length: int) -> float:
    """Return the TM-score distance scale ``d0`` for a chain of ``length`` residues.

    Uses the Zhang & Skolnick (2004) formula implemented in TM-align and biotite:

    ``d0 = max(1.24 * (L - 15)^(1/3) - 1.8, 0.5)``
    """
    if length <= 15:
        return 0.5
    return max(1.24 * ((length - 15) ** (1 / 3)) - 1.8, 0.5)


def _chain_residue_ids(pdb_path: Path) -> list[tuple[str, int]]:
    chain = next(get_structure(str(pdb_path)).get_chains())
    residue_ids: list[tuple[str, int]] = []
    for residue in chain.get_residues():
        if residue.id[0] != " ":
            continue
        if "CA" not in residue.child_dict:
            continue
        residue_ids.append((chain.id, int(residue.id[1])))
    return residue_ids


def compute_per_residue_tm_scores(
    reference_pdb: Path,
    prediction_pdb: Path,
) -> PerResidueTmScoreResult:
    """Compute per-residue local TM scores for the prediction chain.

    Uses ``tmtools.tm_align`` with the reference as chain 1 and the prediction
    as chain 2. Each aligned prediction residue receives
    ``1 / (1 + (d / d0)^2)``; unaligned residues receive ``0``.
    """
    reference_pdb = reference_pdb.expanduser().resolve()
    prediction_pdb = prediction_pdb.expanduser().resolve()

    ref_coords, ref_seq = get_residue_data(next(get_structure(str(reference_pdb)).get_chains()))
    pred_coords, pred_seq = get_residue_data(next(get_structure(str(prediction_pdb)).get_chains()))
    pred_residue_ids = _chain_residue_ids(prediction_pdb)

    if len(pred_residue_ids) != len(pred_seq):
        raise ValueError(
            f"Residue ID count ({len(pred_residue_ids)}) does not match prediction "
            f"sequence length ({len(pred_seq)}) in {prediction_pdb}"
        )

    result = tm_align(ref_coords, pred_coords, ref_seq, pred_seq)
    ref_aligned = ref_coords @ result.u.T + result.t

    d0 = tm_score_d0(len(pred_seq))
    scores = np.zeros(len(pred_seq), dtype=float)
    ref_index = pred_index = 0
    for ref_char, pred_char in zip(result.seqxA, result.seqyA, strict=True):
        if ref_char != "-" and pred_char != "-":
            distance = np.linalg.norm(ref_aligned[ref_index] - pred_coords[pred_index])
            scores[pred_index] = 1.0 / (1.0 + (distance / d0) ** 2)
        if ref_char != "-":
            ref_index += 1
        if pred_char != "-":
            pred_index += 1

    residues = tuple(
        ResidueTmScore(chain_id=chain_id, resi=resi, score=float(score))
        for (chain_id, resi), score in zip(pred_residue_ids, scores, strict=True)
    )
    return PerResidueTmScoreResult(tm_score=float(result.tm_norm_chain2), residues=residues)
