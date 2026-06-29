"""Detect query-masked residue positions by comparing an experiment A3M to the original FASTA."""

from __future__ import annotations

from pathlib import Path

ACID_TOKENS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ-")
ACID_INSERTS = set("abcdefghijklmnopqrstuvwxyz")


def read_fasta_sequence(fasta_path: Path) -> str:
    """Return the first sequence from a FASTA file."""
    lines = [line.strip() for line in fasta_path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"FASTA file is empty: {fasta_path}")
    if lines[0].startswith(">"):
        if len(lines) < 2:
            raise ValueError(f"FASTA file has header but no sequence: {fasta_path}")
        return lines[1]
    return lines[0]


def read_a3m_records(a3m_path: Path) -> list[tuple[str, str]]:
    """Return (header, sequence) pairs from an A3M file."""
    text = a3m_path.read_text()
    if not text.strip():
        raise ValueError(f"A3M file is empty: {a3m_path}")

    records: list[tuple[str, str]] = []
    header: str | None = None
    seq_parts: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_parts)))
            header = line
            seq_parts = []
        elif header is None:
            raise ValueError(f"Expected FASTA header before sequence in {a3m_path}")
        else:
            seq_parts.append(line)

    if header is not None:
        records.append((header, "".join(seq_parts)))

    if not records:
        raise ValueError(f"No records found in A3M file: {a3m_path}")
    return records


def ungap_a3m_sequence(seq: str) -> str:
    """Remove A3M insertion characters and gaps from an aligned sequence."""
    return "".join(char for char in seq if char not in ACID_INSERTS and char != "-")


def read_a3m_query_sequence(a3m_path: Path) -> str:
    """Return the ungapped query sequence (first A3M record)."""
    return ungap_a3m_sequence(read_a3m_records(a3m_path)[0][1])


def find_query_mask_positions(
    original_sequence: str,
    a3m_query_sequence: str,
    *,
    one_based: bool = True,
) -> list[int]:
    """Return residue indices where the A3M query differs from the original sequence."""
    if len(original_sequence) != len(a3m_query_sequence):
        raise ValueError(
            "Original and A3M query sequences have different lengths: "
            f"{len(original_sequence)} vs {len(a3m_query_sequence)}"
        )

    offset = 1 if one_based else 0
    return [
        index + offset
        for index, (original_aa, query_aa) in enumerate(zip(original_sequence, a3m_query_sequence))
        if original_aa != query_aa
    ]


def find_query_mask_positions_from_files(
    fasta_path: Path,
    a3m_path: Path,
    *,
    one_based: bool = True,
) -> list[int]:
    """Compare ``data/<protein>/<protein>.fasta`` to the query row in an experiment A3M."""
    original = read_fasta_sequence(fasta_path)
    a3m_query = read_a3m_query_sequence(a3m_path)
    return find_query_mask_positions(original, a3m_query, one_based=one_based)


def experiment_a3m_path(experiment_dir: Path, protein: str) -> Path:
    """Return the A3M path saved alongside prediction outputs."""
    return experiment_dir / protein / f"{protein}.a3m"
