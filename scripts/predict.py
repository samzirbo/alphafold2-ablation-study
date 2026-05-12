"""
AlphaFold2 structure prediction via ColabFold.

Thin CLI wrapper around colabfold.batch.run() tailored to the project's requirements.

Usage:
    # Run locally
    python scripts/predict.py --job baseline --input LAT1

    # Run on Colab, save to Google Drive (default path)
    python scripts/predict.py --job baseline --input LAT1 --drive

    # Run on Colab, save to a custom Drive path
    python scripts/predict.py --job baseline --drive /content/drive/MyDrive/my-folder/results

    # Override MSA depth
    python scripts/predict.py --job msa_depth_16 --input LAT1 --max-msa 16:32
"""

from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GRPC_VERBOSITY"] = "ERROR"

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn

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

    "keep_existing_results": True,

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


def resolve_proteins(protein: str | None) -> list[str]:
    """Return a list of protein names to predict."""
    if protein is not None:
        protein_dir = DATA_DIR / protein
        if not protein_dir.exists():
            sys.exit(f"Protein directory not found: {protein_dir}")
        if resolve_input_file(protein) is None:
            sys.exit(f"No .a3m or .fasta file found in {protein_dir}")
        return [protein]

    available = []
    missing = []
    for p in PROTEINS:
        if resolve_input_file(p) is not None:
            available.append(p)
        else:
            missing.append(p)

    if missing:
        console.print(f"  [yellow]warning[/] Missing input files for: {missing}")

    if not available:
        console.print("  [red bold]FAILED[/] No proteins found in data/ with .a3m or .fasta files.")
        sys.exit(1)

    return available


def resolve_result_dir(job: str, drive: str | None) -> Path:
    """Return the result directory, mounting Google Drive if requested."""
    if drive is not None:
        from src.utils.drive import get_drive_result_dir
        drive_path = drive if drive != "" else None
        result_dir = get_drive_result_dir(job, drive_path)
    else:
        result_dir = RESULTS_DIR / job

    if result_dir.exists():
        console.print(
            f"  [yellow]warning[/] Result directory already exists: [dim]{result_dir}[/]\n"
            f"          Proteins with a .done.txt marker will be skipped."
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
        default=None,
        help="Protein name (e.g. 'LAT1'). If omitted, runs all available proteins.",
    )
    p.add_argument(
        "--drive",
        nargs="?",
        const="",
        default=None,
        help="Save results to Google Drive. Optionally provide a custom Drive path.",
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

    return p.parse_args(argv)


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

    total_models = args.num_models * args.num_seeds

    console.print()
    for protein in proteins:
        input_path = resolve_input_file(protein)
        queries, is_complex = get_queries(input_path)

        progress = Progress(
            SpinnerColumn(),
            TextColumn(f"  {protein}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("models"),
            console=console,
        )

        with progress:
            task = progress.add_task(protein, total=total_models)

            def prediction_callback(protein_obj, length, prediction_result, input_features, mode):
                progress.advance(task)

            run(
                queries=queries,
                result_dir=result_dir,
                is_complex=is_complex,
                # --- fixed (from RUN_CONFIG) ---
                model_type=model_type,
                random_seed=RUN_CONFIG["random_seed"],
                use_dropout=RUN_CONFIG["use_dropout"],
                use_templates=RUN_CONFIG["use_templates"],
                num_recycles=RUN_CONFIG["num_recycles"],
                recycle_early_stop_tolerance=RUN_CONFIG["recycle_early_stop_tolerance"],
                num_relax=RUN_CONFIG["num_relax"],
                keep_existing_results=RUN_CONFIG["keep_existing_results"],
                user_agent="colabfold/alphafold2-ablation-study",
                # --- overridable (from CLI) ---
                msa_mode=args.msa_mode,
                max_msa=args.max_msa,
                num_models=args.num_models,
                num_seeds=args.num_seeds,
                prediction_callback=prediction_callback,
            )

        console.print(f"  [green]ok[/]        {protein}")

    console.print(f"\n[bold green]Job '{args.job}' complete.[/] Results in [dim]{result_dir}[/]\n")


if __name__ == "__main__":
    main()
