"""
AlphaFold2 structure prediction via ColabFold.

Thin CLI wrapper around colabfold.batch.run() tailored to the project's requirements.

Usage:
    # Run a single protein
    python scripts/predict.py --job baseline --input LAT1

    # Run all proteins in the dataset
    python scripts/predict.py --job baseline

    # Override MSA depth
    python scripts/predict.py --job msa_depth_16 --input LAT1 --max-msa 16:32
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

logger = logging.getLogger(__name__)

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

    "save_all": True,
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
        logger.warning("Missing input files for: %s", missing)

    if not available:
        sys.exit("No proteins found in data/ with .a3m or .fasta files.")

    return available


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
        "--msa-mode",
        default=RUN_CONFIG["msa_mode"],
        choices=["mmseqs2_uniref_env", "custom"],
        help="MSA generation mode. 'custom' uses the precomputed .a3m file.",
    )
    p.add_argument(
        "--max-msa",
        type=str,
        default=RUN_CONFIG["max_msa"],
        help="MSA depth as 'max_seq:max_extra_seq' (e.g. '16:32'). "
        "None uses model default (512:5120).",
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

    result_dir = RESULTS_DIR / args.job
    result_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(result_dir / "log.txt")

    proteins = resolve_proteins(args.input)
    logger.info("Job '%s': proteins=%s", args.job, proteins)
    logger.info(
        "CLI overrides: msa_mode=%s, max_msa=%s, num_models=%d, num_seeds=%d",
        args.msa_mode, args.max_msa, args.num_models, args.num_seeds,
    )

    model_type = set_model_type(False, RUN_CONFIG["model_type"])
    download_alphafold_params(model_type)

    for protein in proteins:
        logger.info("--- Predicting %s ---", protein)
        input_path = resolve_input_file(protein)

        queries, is_complex = get_queries(input_path)

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
            save_all=RUN_CONFIG["save_all"],
            keep_existing_results=RUN_CONFIG["keep_existing_results"],
            user_agent="colabfold/alphafold2-ablation-study",
            # --- overridable (from CLI) ---
            msa_mode=args.msa_mode,
            max_msa=args.max_msa,
            num_models=args.num_models,
            num_seeds=args.num_seeds,
        )

        logger.info("Done: %s", protein)

    logger.info("Job '%s' complete. Results in %s", args.job, result_dir)


if __name__ == "__main__":
    main()
