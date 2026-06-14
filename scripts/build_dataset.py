"""
Build the dataset for the AlphaFold2 ablation study.

Downloads raw FASTA sequences and reference PDB structures
for each protein defined in data/metadata.json.
"""

import argparse
import io
import json
import shutil
import time
from pathlib import Path

import Bio.PDB
from colabfold.colabfold import run_mmseqs2
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


def truncate_sequence(sequence: str, truncation: list[list[int]]) -> str:
    """Remove 1-indexed inclusive residue ranges."""
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

        raw_path = out_dir / f"{name}_raw.fasta"
        fasta_path = out_dir / f"{name}.fasta"

        if fasta_path.exists():
            console.print(f"  [yellow]overwrite[/] {name} ({uid})")

        fasta = fetch_fasta(uid)
        if fasta is None:
            console.print(f"  [red bold]FAILED[/]    {name} ({uid}): download failed")
            raise RuntimeError(f"Failed to fetch FASTA for {name} ({uid})")

        header = fasta.splitlines()[0].strip()
        sequence = "".join(fasta.splitlines()[1:]).strip()
        truncated = truncate_sequence(sequence, info["truncation"])

        seq_len = len(sequence)
        if seq_len != info["sequence_length"]:
            raise ValueError(f"{name}: raw length {seq_len} != expected {info['sequence_length']}")

        raw_path.write_text(f">{name}\n{sequence}\n")
        fasta_path.write_text(f">{name}\n{truncated}\n")

        removed = sum(end - start + 1 for start, end in info["truncation"])
        assert len(truncated) == info["sequence_length"] - removed
        trunc_info = f", truncated {removed} residues" if removed else ""
        console.print(f"  [green]ok[/]        {name} ({uid}) — {seq_len} aa{trunc_info}")

    console.print("  [green]done[/]")


# ---------------------------------------------------------------------------
# PDB structures
# ---------------------------------------------------------------------------

def fetch_pdb(pdb_id: str) -> str | None:
    r = get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb")
    return r.text if r else None


def kept_positions(info: dict) -> set[int]:
    """Return the 1-indexed UniProt positions present in the prediction input."""
    seq_len = info["sequence_length"]
    positions = set(range(1, seq_len + 1))
    for start, end in info["truncation"]:
        positions -= set(range(start, end + 1))
    return positions


def dbref_maps(pdb_text: str, chain: str, uniprot_id: str) -> list[tuple[int, int, int, int]]:
    """
    Return DBREF mappings from PDB residue IDs to target UniProt positions.

    PDB residue numbers are not always the same coordinate system as the input
    sequence. For example, a PDB construct may number target residues as 98-415,
    while UniProt positions are 2-319. DBREF records encode that relationship:

        PDB chain residue range -> database accession residue range

    We keep only DBREF rows for the selected chain and the target UniProt ID
    from metadata. This removes chain segments that belong to ligands, fusion
    partners, or engineered insertions while preserving the target protein. It
    also lets metadata truncations, which are written in UniProt coordinates,
    be applied to references even when the PDB residue numbering is offset.
    """
    maps = []
    for line in pdb_text.splitlines():
        if not line.startswith("DBREF"):
            continue
        parts = line.split()
        if len(parts) >= 10 and parts[2] == chain and parts[5] == "UNP" and parts[6] == uniprot_id:
            maps.append(tuple(map(int, (parts[3], parts[4], parts[8], parts[9]))))
    return maps


def mapped_position(residue_id: int, maps: list[tuple[int, int, int, int]]) -> int | None:
    """
    Map a PDB residue number to its UniProt position using DBREF ranges.

    Returning None means the residue is outside the target UniProt DBREF ranges,
    so it should not be part of the cleaned reference chain.
    """
    for pdb_start, pdb_end, uniprot_start, uniprot_end in maps:
        if pdb_start <= residue_id <= pdb_end:
            step = 1 if uniprot_end >= uniprot_start else -1
            return uniprot_start + step * (residue_id - pdb_start)
    return None


class ReferenceSelect(Bio.PDB.Select):
    """
    PDBIO selector for TM-score-ready reference chains.

    The prediction FASTA is already truncated according to metadata, so the
    reference chain must be trimmed to the same input positions before scoring.
    When target UniProt DBREF records are available, residue filtering happens
    in UniProt coordinates. That removes non-target chain segments such as CCR5
    ligand/fusion residues and makes PDB numbering offsets harmless. When no
    target DBREF exists, we fall back to raw PDB residue IDs
    """

    def __init__(
            self,
            chain_id: str,
            keep_positions: set[int],
            maps: list[tuple[int, int, int, int]]
    ):
        self.chain_id = chain_id
        self.keep_positions = keep_positions
        self.maps = maps

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        # Drop HETATM residues such as ligands, waters, and fusion cofactors.
        # TM-score uses protein C-alpha traces, so keeping only standard ATOM
        # residues prevents non-protein records from entering the reference.
        if residue.id[0] != " ":
            return False

        residue_id = residue.id[1]
        if self.maps:
            # Convert from the PDB author's residue numbering to UniProt
            # numbering before checking metadata truncations. This is what
            # handles offset constructs such as CCR5 active 7F1Q_R.
            position = mapped_position(residue_id, self.maps)
            if position is None:
                return False
        else:
            # Fallback for references without a target UniProt DBREF\
            position = residue_id

        return position in self.keep_positions


def extract_chain(pdb_text: str, chain: str, info: dict) -> str:
    """Extract one cleaned reference chain with Bio.PDB's parser and selector."""
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("ref", io.StringIO(pdb_text))

    available_chains = [c.id for c in structure.get_chains()]
    if chain not in available_chains:
        raise ValueError(
            f"chain {chain} not found; available chains: {sorted(available_chains)}"
        )

    keep = kept_positions(info)
    maps = dbref_maps(pdb_text, chain, info["uniprot_id"])

    out = io.StringIO()
    pdb_io = Bio.PDB.PDBIO()
    pdb_io.set_structure(structure)
    pdb_io.save(out, select=ReferenceSelect(chain, keep, maps))
    chain_data = "\n".join(line.rstrip() for line in out.getvalue().splitlines()) + "\n"

    return chain_data


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
                chain_data = extract_chain(pdb_data, chain, info)
            except ValueError as exc:
                console.print(f"  [red bold]FAILED[/]    {name}/{label}: {exc}")
                raise

            raw_path.write_text(pdb_data)
            chain_path.write_text(chain_data)

            console.print(f"  [green]ok[/]        {name}/{label} — {pdb_id} chain {chain}")

    console.print("  [green]done[/]")


# ---------------------------------------------------------------------------
# MSA (via ColabFold MMseqs2)
# ---------------------------------------------------------------------------

def download_msas(metadata: dict) -> None:
    console.print("\n[bold]Downloading MSAs...[/]")
    for name, info in metadata.items():
        fasta_path = DATA_DIR / name / f"{name}.fasta"
        if not fasta_path.exists():
            raise FileNotFoundError(f"{fasta_path} not found — run sequence download first")

        out_path = DATA_DIR / name / f"{name}.a3m"
        if out_path.exists():
            console.print(f"  [yellow]overwrite[/] {name}")

        query_seq = fasta_path.read_text().splitlines()[1].strip()
        prefix = str(DATA_DIR / name / "mmseqs2")
        mmseqs2_dir = Path(f"{prefix}_env")

        a3m = run_mmseqs2(
            query_seq,
            prefix,
            user_agent="alphafold2-ablation-study",
        )[0]

        first_seq = a3m.splitlines()[1].strip()
        if first_seq != query_seq:
            console.print(f"  [red bold]MISMATCH[/] {name}: first MSA sequence != query")
            raise ValueError(f"{name}: MSA query sequence mismatch")

        depth = a3m.count("\n>")
        out_path.write_text(a3m)

        if mmseqs2_dir.exists():
            shutil.rmtree(mmseqs2_dir)

        console.print(f"  [green]ok[/]        {name} — {depth} sequences")

    console.print("  [green]done[/]")

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
    download_msas(metadata)


if __name__ == "__main__":
    main()
