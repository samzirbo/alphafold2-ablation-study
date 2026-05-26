from pathlib import Path
import numpy as np
import tempfile
from rich.console import Console

console = Console()

TMP_DIR: Path | None = None

ACID_TOKENS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ-")
ACID_INSERTS = set("abcdefghijklmnopqrstuvwxyz")

def get_tmp_dir() -> Path:
    global TMP_DIR
    if TMP_DIR is None:
        TMP_DIR = Path(tempfile.mkdtemp(prefix="a3m_ablations_"))
    return TMP_DIR

def random_row_column_masking(a3m_file: Path, seed=0, frac=0.1, row_or_col="column", out_file: Path | None = None) -> Path:
    """
    Creates a row- or column-masked a3m file.

    Args:
        a3m_file (Path):            File to be masked.
        seed (int):                 Seed for the random masking.
        frac (float):               Fraction of rows or columns to be masked.
        row_or_col (str):           Switch to select row or column masking.
        out-file (Path | None):     Path where the masked file should be written to.
                                    Written to a temporary file if None.

    Returns:
        Path: Path to the masked file.
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError("frac should be between 0.0 and 1.0")
    rng = np.random.default_rng(seed)

    lines = a3m_file.read_text().splitlines()
    records = []
    for i in range(0, len(lines), 2):
        if lines[i].startswith(">"):
            records.append([lines[i], lines[i + 1].strip()])
        else:
            console.print(f"[red bold] Expected a header line starting with \">\" at position {i}. Instead line was {lines[i]}[/]")
            raise RuntimeError(f"Failed to create {row_or_col}-masked a3m file")

    if row_or_col == "column":
        n_cols = len(records[0][1])
        rand_cols = set(rng.choice(n_cols, int(n_cols * frac), replace=False))

        for i in range(1, len(records)):
            seq = list(records[i][1])
            j = -1
            for k, char in enumerate(seq):
                if (char in ACID_TOKENS):
                    j += 1
                    if j in rand_cols:
                        seq[k] = "X"
                elif not (char in ACID_INSERTS): 
                    console.print(f"[red bold] Unexpected character in aligned fasta sequence {i} at position {k}: {char}[/]")
                    raise RuntimeError("Failed to create column-masked a3m file")
            if j+1 != len(records[0][1]):
                console.print(f"[red bold] Aligned sequence length ({j+1}) in line {(i+1)*2} does not match input sequence length ({len(records[0][1])})[/]")
                raise RuntimeError("Failed to create column-masked a3m file")
            records[i][1] = "".join(seq)

    elif row_or_col == "row":
        n_rows = len(records)
        rand_rows = set(rng.choice(range(1, n_rows), int((n_rows - 1) * frac), replace=False))

        for i in range(1, n_rows):
            if i in rand_rows:
                seq = list(records[i][1])
                j = -1
                for k, char in enumerate(seq):
                    if (char in ACID_TOKENS):
                        j += 1
                        seq[k] = "X"
                    elif not (char in ACID_INSERTS):
                        console.print(f"[red bold] Unexpected character in aligned fasta sequence {i} at position {k}: {char}[/]")
                        raise RuntimeError("Failed to create row-masked a3m file")
                if j+1 != len(records[0][1]):
                    console.print(f"[red bold] Aligned sequence length ({j+1}) in line {(i+1)*2} does not match input sequence length ({len(records[0][1])})[/]")
                    raise RuntimeError("Failed to create row-masked a3m file")
                records[i][1] = "".join(seq)
    
    else:
        raise ValueError("row_or_col should be \"column\" or \"row\".")

    if out_file is None:
        out_file = get_tmp_dir() / a3m_file.name
    else:
        out_file = Path(out_file)

    out_file.write_text("\n".join(f"{h}\n{s}" for h, s in records) + "\n")
    console.print(f"[green]Wrote {row_or_col}-masked file {out_file.name}[/]")

    return out_file


def row_masking(a3m_file: Path, n_keep: int, out_file: Path | None = None) -> Path:
    """
    Creates an a3m file where all sequences beyond the first n_keep hits are masked.

    Args:
        a3m_file (Path):        File to be masked.
        n_keep (int):           Number of hit sequences to keep unmasked.
        out_file (Path | None): Path where the masked file should be written to. 
                                Written to a temporary file if None.   

    Returns:
        Path: Path to the masked file.
    """
    if n_keep < 0:
        raise ValueError("n_keep must be non-negative")

    lines = a3m_file.read_text().splitlines()
    records = []
    for i in range(0, len(lines), 2):
        if lines[i].startswith(">"):
            records.append([lines[i], lines[i + 1].strip()])
        else:
            console.print(f"[red bold] Expected a header line starting with \">\" at position {i}. Instead line was {lines[i]}[/]")
            raise RuntimeError("Failed to create row-masked a3m file")

    n_hits = len(records) - 1
    if n_keep >= n_hits:
        console.print(f"[yellow]warning[/] n_keep ({n_keep}) >= number of hit sequences ({n_hits}) — nothing to mask")

    for i in range(n_keep + 1, len(records)):
        seq = list(records[i][1])
        j = -1
        for k, char in enumerate(seq):
            if char in ACID_TOKENS:
                j += 1
                seq[k] = "X"
            elif char not in ACID_INSERTS:
                console.print(f"[red bold] Unexpected character in aligned fasta sequence {i} at position {k}: {char}[/]")
                raise RuntimeError("Failed to create row-masked a3m file")
        if j + 1 != len(records[0][1]):
            console.print(f"[red bold] Aligned sequence length ({j+1}) in line {(i+1)*2} does not match input sequence length ({len(records[0][1])})[/]")
            raise RuntimeError("Failed to create row-masked a3m file")
        records[i][1] = "".join(seq)

    if out_file is None:
        out_file = get_tmp_dir() / a3m_file.name
    else:
        out_file = Path(out_file)

    out_file.write_text("\n".join(f"{h}\n{s}" for h, s in records) + "\n")
    console.print(f"[green]Wrote row-masked file {out_file.name}[/]")

    return out_file


def random_query_masking(a3m_file: Path, frac: float, seed: int = 7, out_file: Path | None = None) -> Path:
    """
    Creates an a3m file where a fraction of amino acids in the query sequence are masked and returns the path to it.

    Only the query (first) sequence is modified; all MSA hit sequences remain untouched.
    Masking replaces selected amino acids with the token ``"X"``.

    **Maskable positions** are all characters in the query that are valid amino-acid
    tokens (uppercase ``A-Z``), *excluding* gap characters (``"-"``) and any
    pre-existing mask tokens (``"X"``).  The ``frac`` parameter specifies what
    fraction of these maskable positions will be replaced with ``"X"``.

    Which specific positions are masked is chosen uniformly at random, controlled
    by the ``seed`` parameter for reproducibility.

    Args:
        a3m_file (Path):        File to be masked.
        frac (float):           Required. Fraction of maskable tokens (amino acids
                                excluding ``"-"`` and ``"X"``) in the query sequence
                                to replace with ``"X"``. Must be between 0.0 and 1.0.
        seed (int):             Seed for the random number generator. Default is 7.
        out_file (Path | None): Path where the masked file should be written to.
                                Written to a temporary file if None.

    Returns:
        Path: Path to the masked file.
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError("frac should be between 0.0 and 1.0")

    rng = np.random.default_rng(seed)

    lines = a3m_file.read_text().splitlines()
    records = []
    for i in range(0, len(lines), 2):
        if lines[i].startswith(">"):
            records.append([lines[i], lines[i + 1].strip()])
        else:
            console.print(f'[red bold] Expected a header line starting with ">" at position {i}. Instead line was {lines[i]}[/]')
            raise RuntimeError("Failed to create query-masked a3m file")

    # Identify maskable positions: amino-acid tokens that are NOT "-" or "X"
    query_seq = list(records[0][1])
    maskable_indices = [
        k for k, char in enumerate(query_seq)
        if char in ACID_TOKENS and char not in ("-", "X")
    ]

    n_mask = int(len(maskable_indices) * frac)
    if n_mask > 0:
        chosen = rng.choice(maskable_indices, n_mask, replace=False)
        for k in chosen:
            query_seq[k] = "X"

    records[0][1] = "".join(query_seq)

    if out_file is None:
        out_file = get_tmp_dir() / a3m_file.name
    else:
        out_file = Path(out_file)

    out_file.write_text("\n".join(f"{h}\n{s}" for h, s in records) + "\n")
    console.print(f"[green]Wrote query-masked file {out_file.name} ({n_mask}/{len(maskable_indices)} positions masked)[/]")

    return out_file


# CLI-exposed ablations. Keys are the function name; "params" maps each
# function kwarg to (type, default). `None` default means the kwarg is required.
ABLATIONS = {
    "row_or_col_masking": {
        "fn": random_row_column_masking,
        "params": {
            "seed":       (int,   0),
            "frac":       (float, 0.1),
            "row_or_col": (str,   "column"),
        },
    },
    "row_masking": {
        "fn": row_masking,
        "params": {
            "n_keep": (int, None),
        },
    },
    "query_masking": {
        "fn": random_query_masking,
        "params": {
            "frac": (float, None),
            "seed": (int,   7),
        },
    },
}