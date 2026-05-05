"""
Build the dataset for the AlphaFold2 ablation study.

Downloads raw FASTA sequences and reference PDB structures
for each protein defined in data/metadata.json.
"""

import argparse
import json
import shutil
import textwrap
import time
from pathlib import Path

import requests
from rich.console import Console

console = Console()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
METADATA_PATH = DATA_DIR / "metadata.json"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get(url: str, retries: int = 3, **kwargs) -> requests.Response | None:
    kwargs.setdefault("timeout", 30)
    for attempt in range(retries):
        try:
            r = requests.get(url, **kwargs)
            if r.status_code == 200:
                return r
            console.print(f"  [yellow]HTTP {r.status_code}[/] from [dim]{url}[/]")
        except requests.RequestException as exc:
            console.print(f"  [yellow]retry {attempt + 1}/{retries}[/]: {exc}")
        time.sleep(1)
    return None


def clean(metadata: dict) -> None:
    console.print("\n[bold]Cleaning dataset...[/]")
    for name in metadata:
        target_dir = DATA_DIR / name
        if target_dir.exists():
            shutil.rmtree(target_dir)
            console.print(f"  [dim]removed[/] {name}/")
    console.print("  [green]done[/]\n")


def load_metadata() -> dict:
    with open(METADATA_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# FASTA
# ---------------------------------------------------------------------------

def fetch_fasta(uniprot_id: str) -> str | None:
    r = get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta")
    return r.text if r else None


def extract_sequence(fasta: str) -> str:
    return "".join(
        line.strip()
        for line in fasta.splitlines()
        if line and not line.startswith(">")
    )


def truncate_sequence(fasta: str, truncation: list[list[int]]) -> str:
    """Remove 1-indexed inclusive residue ranges."""
    sequence = extract_sequence(fasta)

    if not truncation:
        return sequence

    remove = set()
    for start, end in truncation:
        remove.update(range(start, end + 1))

    return "".join(
        residue
        for i, residue in enumerate(sequence, start=1)
        if i not in remove
    )


def download_sequences(metadata: dict) -> None:
    console.print("\n[bold]Downloading sequences...[/]")
    for name, info in metadata.items():
        uid = info["uniprot_id"]
        out_dir = DATA_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_path = out_dir / "sequence_raw.fasta"
        truncated_path = out_dir / "sequence_truncated.fasta"

        if raw_path.exists():
            console.print(f"  [yellow]overwrite[/] {name} ({uid})")

        fasta = fetch_fasta(uid)
        if fasta is None:
            console.print(f"  [red bold]FAILED[/]    {name} ({uid}): download failed")
            raise RuntimeError(f"Failed to fetch FASTA for {name} ({uid})")

        header = fasta.splitlines()[0]
        sequence = extract_sequence(fasta)
        truncated = header + "\n" + textwrap.fill(
            truncate_sequence(fasta, info["truncation"]), width=60
        )

        seq_len = len(sequence)
        if seq_len != info["sequence_length"]:
            raise ValueError(f"{name}: raw length {seq_len} != expected {info['sequence_length']}")

        raw_path.write_text(fasta)
        truncated_path.write_text(truncated)

        removed = sum(end - start + 1 for start, end in info["truncation"])
        trunc_info = f", truncated {removed} residues" if removed else ""
        console.print(f"  [green]ok[/]        {name} ({uid}) — {seq_len} aa{trunc_info}")

    console.print("  [green]done[/]")


# ---------------------------------------------------------------------------
# PDB structures
# ---------------------------------------------------------------------------

def fetch_pdb(pdb_id: str) -> str | None:
    r = get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb")
    return r.text if r else None


def extract_chain(pdb_text: str, chain: str) -> str:
    """Keep only ATOM/HETATM/TER records for the given chain, plus END."""
    lines = []
    atom_count = 0
    available_chains = set()

    for line in pdb_text.splitlines():
        record = line[:6].strip()

        if record in ("ATOM", "HETATM", "TER"):
            if len(line) > 21:
                current_chain = line[21]
                available_chains.add(current_chain)

                if current_chain == chain:
                    lines.append(line)
                    if record in ("ATOM", "HETATM"):
                        atom_count += 1

        elif record == "END":
            continue

    if atom_count == 0:
        raise ValueError(
            f"chain {chain} not found; available chains: {sorted(available_chains)}"
        )

    lines.append("END")
    return "\n".join(lines) + "\n"


def download_structures(metadata: dict) -> None:
    console.print("\n[bold]Downloading structures...[/]")
    for name, info in metadata.items():
        references_dir = DATA_DIR / name / "references"
        references_dir.mkdir(parents=True, exist_ok=True)

        for conf in info["conformations"].values():
            label = conf["label"]
            pdb_id = conf["pdb_id"].upper()
            chain = conf["chain"]

            raw_path = references_dir / f"{label}_{pdb_id}.pdb"
            chain_path = references_dir / f"{label}_{pdb_id}_{chain}.pdb"

            if raw_path.exists() or chain_path.exists():
                console.print(f"  [yellow]overwrite[/] {name}/{label}")

            pdb_data = fetch_pdb(pdb_id)
            if pdb_data is None:
                console.print(f"  [red bold]FAILED[/]    {name}/{label}: download failed")
                raise RuntimeError(f"Failed to download PDB {pdb_id} for {name}/{label}")

            try:
                chain_data = extract_chain(pdb_data, chain)
            except ValueError as exc:
                console.print(f"  [red bold]FAILED[/]    {name}/{label}: {exc}")
                raise

            raw_path.write_text(pdb_data)
            chain_path.write_text(chain_data)

            console.print(f"  [green]ok[/]        {name}/{label} — {pdb_id} chain {chain}")

    console.print("  [green]done[/]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build the AlphaFold2 ablation study dataset."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove all downloaded data and re-download.",
    )
    args = parser.parse_args()

    metadata = load_metadata()

    if args.clean:
        clean(metadata)

    download_sequences(metadata)
    download_structures(metadata)


if __name__ == "__main__":
    main()
