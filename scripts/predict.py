"""
AlphaFold2 structure prediction via ColabFold.

Thin CLI wrapper around colabfold.batch.run() tailored to the project's requirements.
Each (model, seed) combination is run individually so that completed predictions
are automatically skipped on restart.

Usage:
    # Run a single protein
    python scripts/predict.py --job baseline --input LAT1

    # Run multiple proteins
    python scripts/predict.py --job baseline --input LAT1 ZnT8 MCT1

    # Run all available proteins
    python scripts/predict.py --job baseline

    # Save results to a custom path (e.g. Google Drive on Colab)
    python scripts/predict.py --job baseline --input LAT1 --drive /content/drive/MyDrive/results

    # Override MSA depth
    python scripts/predict.py --job msa_depth_16 --input LAT1 --max-msa 16:32
"""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GRPC_VERBOSITY"] = "ERROR"

import argparse
import json
import shutil
import sys
import tempfile
from itertools import product
from pathlib import Path

import time # for the FUSE fix (folder duplicates)

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

console = Console()

PROTEINS = [
    "LAT1", "ZnT8", "MCT1", "STP10", "ASCT2",
    "CGRPR", "PTH1R", "FZD7",
    "SERT", "PfMATE", "MurJ", "CCR5",
]

RUN_CONFIG = {
    # FIXED PARAMETERS
    "model_type": "alphafold2",
    "random_seed": 0, # for reproducibility
    "use_dropout": False,

    "use_templates": False,

    "num_recycles": 1,
    "recycle_early_stop_tolerance": 0.0, # avoid early stopping

    "num_relax": 0,

    # DEFAULT PARAMETERS - can be overridden by CLI
    "msa_mode": "custom",
    "max_msa": "512:5120",

    "num_models": 5, # 5 models per seed
    "num_seeds": 5, # 25 configurations
}


def resolve_input_file(protein: str) -> Path | None:
    """Return the input file for a protein: prefer .a3m, fall back to .fasta."""
    protein_dir = DATA_DIR / protein
    a3m = protein_dir / f"{protein}.a3m"
    fasta = protein_dir / f"{protein}.fasta"

    if a3m.exists():
        return a3m
    if fasta.exists():
        return fasta
    return None


def resolve_proteins(proteins: list[str] | None) -> list[str]:
    """Return a list of protein names to predict."""
    candidates = proteins if proteins is not None else PROTEINS

    available = []
    missing = []
    for p in candidates:
        protein_dir = DATA_DIR / p
        if not protein_dir.exists():
            missing.append(p)
        elif resolve_input_file(p) is None:
            missing.append(p)
        else:
            available.append(p)

    if missing and proteins is None:
        console.print(f"  [yellow]warning[/] Missing input files for: {missing}")

    if not available:
        console.print("  [red bold]FAILED[/] No proteins found in data/ with .a3m or .fasta files.")
        sys.exit(1)

    return available


def resolve_result_dir(job: str, drive: str | None) -> Path:
    """Return the result directory: --drive path if given, otherwise results/<job>/."""
    result_dir = Path(drive) / job if drive else RESULTS_DIR / job

    if result_dir.exists():
        console.print(
            f"  [yellow]warning[/] Result directory already exists: [dim]{result_dir}[/]\n"
            f"          Existing model/seed predictions will be skipped."
        )
    else:
        result_dir.mkdir(parents=True)

    return result_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--job",
        type=str,
        required=True,
        help="Name for this prediction run. Results saved to results/<job>/.",
    )
    p.add_argument(
        "--input",
        type=str,
        nargs="+",
        default=None,
        help="One or more protein names (e.g. 'LAT1 ZnT8'). If omitted, runs all available proteins.",
    )
    p.add_argument(
        "--drive",
        type=str,
        default=None,
        help="Save results to an external path (e.g. a mounted Google Drive directory).",
    )

    p.add_argument(
        "--msa-mode",
        default=RUN_CONFIG["msa_mode"],
        choices=["mmseqs2_uniref_env", "custom"],
        help="MSA generation mode. 'custom' uses the precomputed .a3m file.",
    )
    p.add_argument(
        "--max-msa",
        type=str,
        default=RUN_CONFIG["max_msa"],
        help="MSA depth as 'max_seq:max_extra_seq' (e.g. '16:32').",
    )
    p.add_argument(
        "--num-models",
        type=int,
        default=RUN_CONFIG["num_models"],
        choices=[1, 2, 3, 4, 5],
        help="Number of model checkpoints to use.",
    )
    p.add_argument(
        "--num-seeds",
        type=int,
        default=RUN_CONFIG["num_seeds"],
        help="Seeds per model checkpoint. Total = num_models × num_seeds.",
    )

    p.add_argument(
        "--qm-frac",
        type=float,
        default=None,
        help="Fraction of the query sequence to mask with 'X' (0.0 to 1.0).",
    )
    p.add_argument(
        "--qm-seed",
        type=int,
        default=7,
        help="Random seed for query masking.",
    )

    return p.parse_args(argv)


def prediction_exists(result_dir: Path, protein: str, model_num: int, seed: int) -> bool:
    """Check whether a prediction PDB already exists for this model/seed."""
    tag = f"alphafold2_model_{model_num}_seed_{seed:03d}"
    return any(result_dir.glob(f"{protein}_unrelaxed_*_{tag}.pdb"))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from colabfold.batch import get_queries, run, set_model_type
    from colabfold.download import download_alphafold_params
    from colabfold.utils import setup_logging

    result_dir = resolve_result_dir(args.job, args.drive)
    setup_logging(result_dir / "log.txt")

    proteins = resolve_proteins(args.input)

    console.print(f"\n[bold]Running job: {args.job}[/]")
    console.print(f"  proteins: {', '.join(proteins)}")
    console.print(f"  results:  [dim]{result_dir}[/]")

    cli_params = {"msa_mode": args.msa_mode, "max_msa": args.max_msa,
                  "num_models": args.num_models, "num_seeds": args.num_seeds}
    overrides = {k: v for k, v in cli_params.items() if v != RUN_CONFIG[k]}

    if overrides:
        console.print(f"  [yellow]overrides[/]: {overrides}")

    model_type = set_model_type(False, RUN_CONFIG["model_type"])
    download_alphafold_params(model_type)

    models = list(range(1, args.num_models + 1))
    seeds = list(range(RUN_CONFIG["random_seed"], RUN_CONFIG["random_seed"] + args.num_seeds))
    total_per_protein = len(models) * len(seeds)

    console.print()
    for protein in proteins:
        input_path = resolve_input_file(protein)

        if args.qm_frac is not None:
            from scripts.ablations import random_query_masking
            console.print(f"  [blue]info[/] Applying query masking to {protein} (frac={args.qm_frac}, seed={args.qm_seed})")
            input_path = random_query_masking(input_path, frac=args.qm_frac, seed=args.qm_seed)

        queries, is_complex = get_queries(input_path)

        protein_result_dir = result_dir / protein
        # Workaround for Google Drive FUSE latency: avoid parents=True to prevent
        # creating duplicate parent directories if FUSE cache is lagging.
        for _ in range(5):
            try:
                protein_result_dir.mkdir(exist_ok=True)
                break
            except FileNotFoundError:
                time.sleep(2)
        else:
            protein_result_dir.mkdir(parents=True, exist_ok=True)

        skipped = 0

        progress = Progress(
            SpinnerColumn(),
            TextColumn(f"  {protein}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("models"),
            console=console,
        )

        with progress:
            task = progress.add_task(protein, total=total_per_protein)

            for i, (model_num, seed) in enumerate(product(models, seeds)):
                if prediction_exists(protein_result_dir, protein, model_num, seed):
                    skipped += 1
                    console.print(f"  [blue]skipped[/]    {protein} model_{model_num} seed_{seed:03d}")
                    progress.advance(task)
                    continue

                # Write to a local staging dir to avoid Google Drive FUSE
                # rename race conditions, then copy outputs to the final dir.
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir)
                    try:
                        run(
                            queries=queries,
                            result_dir=tmp_path,
                            is_complex=is_complex,
                            model_type=model_type,
                            # -- fixed as we are only running one model per seed --
                            num_models=1,
                            model_order=[model_num],
                            num_seeds=1,
                            random_seed=seed,
                            # --- fixed ---
                            use_dropout=RUN_CONFIG["use_dropout"],
                            use_templates=RUN_CONFIG["use_templates"],
                            num_recycles=RUN_CONFIG["num_recycles"],
                            recycle_early_stop_tolerance=RUN_CONFIG["recycle_early_stop_tolerance"],
                            num_relax=RUN_CONFIG["num_relax"],
                            keep_existing_results=False, # we have our own way of tracking existing predictions
                            user_agent="colabfold/alphafold2-ablation-study",
                            # --- overridable ---
                            msa_mode=args.msa_mode,
                            max_msa=args.max_msa,
                            skip_output=["plots"] if i == 0 else ["plots", "msa"],
                        )

                        for f in tmp_path.glob("*.pdb"):
                            shutil.copy2(f, protein_result_dir / f.name)
                        for f in tmp_path.glob("*scores*.json"):
                            shutil.copy2(f, protein_result_dir / f.name)

                        # Save one msa and config file per protein
                        if i == 0:
                            a3m = tmp_path / f"{protein}.a3m"
                            if a3m.exists():
                                shutil.copy2(a3m, protein_result_dir / a3m.name)
                            config = tmp_path / "config.json"
                            if config.exists():
                                shutil.copy2(config, protein_result_dir / config.name)

                        console.print(f"  [green]finished[/]        {protein} model_{model_num} seed_{seed:03d}")
                    except Exception as e:
                        console.print(
                            f"  [red bold]FAILED[/] model_{model_num} seed_{seed:03d}: {e}"
                        )

                progress.advance(task)

        if skipped:
            console.print(
                f"  [green]runs completed[/]        {protein}  "
                f"[dim]({skipped}/{total_per_protein} skipped)[/]"
            )
        else:
            console.print(f"  [green]runs completed[/]        {protein}")

    console.print(
        f"\n[bold green]Job '{args.job}' complete.[/] Results in [dim]{result_dir}[/]\n"
    )


if __name__ == "__main__":
    main()
