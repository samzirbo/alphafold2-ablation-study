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

    # Run with explicit input file paths (instead of names under data/)
    python scripts/predict.py --job custom --input-files /path/to/LAT1.a3m /path/to/ZnT8.fasta

    # Save results to a custom path (e.g. Google Drive on Colab)
    python scripts/predict.py --job baseline --input LAT1 --drive /content/drive/MyDrive/results

    # Override MSA depth
    python scripts/predict.py --job msa_depth_16 --input LAT1 --max-msa 16:32

        # Apply one or more ablations to the input .a3m (daisy-chained, left-to-right)
    python scripts/predict.py --job col_mask_10 --input LAT1 \\
        --ablation row_or_col_masking:row_or_col=column,frac=0.1,seed=0
    python scripts/predict.py --job keep50_then_col10 --input LAT1 \\
        --ablation row_masking:n_keep=50 \\
        --ablation row_or_col_masking:row_or_col=column,frac=0.1,seed=0

    # Apply an ablation to the input .a3m before prediction.
    # Format: --ablation NAME:arg1=value1,arg2=value2
    #
    # Available ablations (defined in scripts/ablations.py):
    #   row_or_col_masking  – mask random MSA rows or columns
    #       params: row_or_col (str, default "column"), frac (float, default 0.1), seed (int, default 0)
    #   row_masking         – mask all MSA rows beyond the first n_keep hits
    #       params: n_keep (int, required)
    #   query_masking       – mask a fraction of amino acids in the query sequence
    #       params: frac (float, required), seed (int, default 7)
    #
    # Parameters with defaults can be omitted. For example, these two are equivalent:
    #   --ablation query_masking:frac=0.3,seed=7
    #   --ablation query_masking:frac=0.3
    #
    # Multiple --ablation flags are daisy-chained left-to-right: the output of
    # the first ablation becomes the input to the next.
    #
    # IMPORTANT: Use a distinct --job name when changing ablations, since the
    # prediction-skip check only keys on (protein, model, seed).

    # Simple: mask 30% of the query sequence
    python scripts/predict.py --job qm_30 --input LAT1 \\
        --ablation query_masking:frac=0.3

    # Complex: keep only top-50 MSA hits, then mask 10% of MSA columns,
    # then mask 20% of the query sequence (all daisy-chained)
    python scripts/predict.py --job keep50_col10_qm20 --input LAT1 \\
        --ablation row_masking:n_keep=50 \\
        --ablation row_or_col_masking:row_or_col=column,frac=0.1,seed=0 \\
        --ablation query_masking:frac=0.2
"""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GRPC_VERBOSITY"] = "ERROR"

import argparse
import json
import logging
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

from ablations import ABLATIONS, get_tmp_dir

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


def resolve_input_paths(paths: list[Path]) -> list[Path]:
    """Validate user-provided input file paths (.a3m or .fasta) and ensure unique stems."""
    valid = []
    for p in paths:
        if not p.exists():
            console.print(f"  [yellow]warning[/] Input file not found: [dim]{p}[/]")
            continue
        if p.suffix not in (".a3m", ".fasta"):
            console.print(
                f"  [yellow]warning[/] Skipping [dim]{p}[/]: unsupported suffix (need .a3m or .fasta)"
            )
            continue
        valid.append(p)

    stems = [p.stem for p in valid]
    duplicates = sorted({s for s in stems if stems.count(s) > 1})
    if duplicates:
        console.print(
            f"  [red bold]FAILED[/] Multiple input files share the same name "
            f"(would collide in the output directory): {duplicates}"
        )
        sys.exit(1)

    if not valid:
        console.print("  [red bold]FAILED[/] No valid input files provided.")
        sys.exit(1)

    return valid


def resolve_result_dir(job: str, drive: str | None) -> Path:
    """Return the result directory: --drive path if given, otherwise results/<job>/.

    The returned path is resolved (symlinks followed) so that subsequent
    operations (mkdir for protein sub-dirs, file copies) work on the
    canonical filesystem path.  This is critical when the Drive path
    passes through a Google Drive shortcut (symlink to a shared drive):
    FUSE can open/write files through the shortcut, but mkdir for child
    directories silently fails with ENOENT.
    """
    result_dir = Path(drive) / job if drive else RESULTS_DIR / job

    if result_dir.exists():
        console.print(
            f"  [yellow]warning[/] Result directory already exists: [dim]{result_dir}[/]\n"
            f"          Existing model/seed predictions will be skipped."
        )
    else:
        result_dir.mkdir(parents=True)
        # Workaround for Google Drive FUSE (avoid duplicate folders): Wait until the directory is visible to the cache
        for _ in range(15):
            if result_dir.exists():
                break
            # Force FUSE cache refresh
            if result_dir.parent.exists():
                try:
                    os.listdir(result_dir.parent)
                except Exception:
                    pass
            time.sleep(1)

    # Resolve symlinks (Google Drive shortcuts to shared drives) so that
    # all subsequent child-directory creation uses the real path.
    resolved = result_dir.resolve()
    if resolved != result_dir:
        console.print(f"  [dim]resolved Drive shortcut → {resolved}[/]")
    return resolved


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
    input_group = p.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input",
        type=str,
        nargs="+",
        default=None,
        help="One or more protein names (e.g. 'LAT1 ZnT8'). If omitted, runs all available proteins.",
    )
    input_group.add_argument(
        "--input-files",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit paths to .a3m or .fasta input files. Output subdir uses each file's stem.",
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

    ablation_help = (
        "Apply an ablation from scripts/ablations.py to the input .a3m before "
        "prediction. Format: NAME:k=v,k=v. Repeatable; ablations are daisy-chained "
        "in the order given. Available: "
        + "; ".join(
            f"{name}({', '.join(params['params'])})"
            for name, params in ABLATIONS.items()
        )
        + ". Use a distinct --job name when changing ablations, since the "
        "prediction-skip check only keys on (protein, model, seed)."
    )
    p.add_argument(
        "--ablation",
        action="append",
        default=[],
        metavar="NAME:k=v,k=v",
        help=ablation_help,
    )

    return p.parse_args(argv)


def parse_ablation_spec(spec: str) -> tuple[str, dict]:
    """Parse 'name:k=v,k=v' into (name, typed_kwargs)."""
    name, _, rest = spec.partition(":")
    if name not in ABLATIONS:
        raise ValueError(f"Unknown ablation '{name}'. Available: {list(ABLATIONS)}")
    schema = ABLATIONS[name]["params"]
    kwargs = {}
    if rest:
        for pair in rest.split(","):
            k, _, v = pair.partition("=")
            k, v = k.strip(), v.strip()
            if k not in schema:
                raise ValueError(f"Unknown param '{k}' for ablation '{name}'. Valid: {list(schema)}")
            type_, _ = schema[k]
            kwargs[k] = type_(v)
    for k, (_, default) in schema.items():
        if default is None and k not in kwargs:
            raise ValueError(f"Ablation '{name}' requires '{k}=...'")
    return name, kwargs


def apply_ablations(input_path: Path, ablations: list[tuple[str, dict]]) -> Path:
    """Daisy-chain ablations on input_path. All steps write to a single tmp file."""
    if not ablations:
        return input_path
    out_file = get_tmp_dir() / input_path.name
    current = input_path
    for name, kwargs in ablations:
        current = ABLATIONS[name]["fn"](current, out_file=out_file, **kwargs)
    return current


def prediction_exists(result_dir: Path, protein: str, model_num: int, seed: int) -> bool:
    """Check whether a prediction PDB already exists for this model/seed."""
    tag = f"alphafold2_model_{model_num}_seed_{seed:03d}"
    return any(result_dir.glob(f"{protein}_unrelaxed_*_{tag}.pdb"))


def _safe_copy(src: Path, dst: Path) -> None:
    """Copy *src* to *dst* via a temporary file to avoid partial writes.

    Writes to ``<dst>.tmp`` first, then renames.  This ensures that
    ``prediction_exists()`` never sees a half-written PDB file, so a
    crashed run can be safely resumed.
    """
    tmp_dst = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp_dst)
    tmp_dst.rename(dst)


def _ensure_drive_dir(target_dir: Path, parent_dir: Path) -> None:
    """Create *target_dir* with FUSE-aware retries.

    Google Drive FUSE can lag in propagating newly created parent directories,
    causing ``mkdir`` to fail with ``ENOENT``.  This function retries with
    cache-refresh tricks and falls back to re-resolving the parent path.
    """
    if target_dir.exists():
        return

    for _attempt in range(20):
        try:
            # Force FUSE cache refresh on the parent directory
            try:
                os.listdir(parent_dir)
            except Exception:
                pass
            target_dir.mkdir(exist_ok=True)
            return
        except FileNotFoundError:
            time.sleep(2)

    # Fallback: re-resolve the parent in case FUSE remounted
    try:
        resolved_parent = parent_dir.resolve()
        resolved_dir = resolved_parent / target_dir.name
        resolved_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"  [dim]created dir via re-resolved path: {resolved_dir}[/]")
        return
    except Exception:
        pass

    console.print(f"  [red bold]FAILED[/] Could not create directory: {target_dir}")
    sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    try:
        ablations = [parse_ablation_spec(s) for s in args.ablation]
    except ValueError as e:
        console.print(f"  [red bold]FAILED[/] {e}")
        sys.exit(1)

    from colabfold.batch import get_queries, run, set_model_type
    from colabfold.download import download_alphafold_params
    from colabfold.utils import setup_logging

    result_dir = resolve_result_dir(args.job, args.drive)
    setup_logging(result_dir / "log.txt")

    # ── Write job-level header to log.txt ──
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("JOB: %s", args.job)
    # Log the raw CLI input args (before input_paths is resolved below)
    if args.input_files is not None:
        logger.info("  input-files: %s", " ".join(str(p) for p in args.input_files))
    elif args.input is not None:
        logger.info("  input:      %s", " ".join(args.input))
    else:
        logger.info("  input:      (all available proteins)")
    logger.info("  max-msa:    %s", args.max_msa)
    logger.info("  msa-mode:   %s", args.msa_mode)
    logger.info("  num-models: %d", args.num_models)
    logger.info("  num-seeds:  %d", args.num_seeds)
    if args.drive:
        logger.info("  drive:      %s", args.drive)
    if args.ablation:
        for spec in args.ablation:
            logger.info("  ablation:   %s", spec)
    else:
        logger.info("  ablations:  (none)")
    logger.info("=" * 60)

    if args.input_files is not None:
        input_paths = resolve_input_paths(args.input_files)
    else:
        input_paths = [resolve_input_file(p) for p in resolve_proteins(args.input)]

    if ablations:
        non_a3m = [p for p in input_paths if p.suffix != ".a3m"]
        if non_a3m:
            console.print(
                f"  [red bold]FAILED[/] --ablation requires .a3m input, got: "
                f"{[str(p) for p in non_a3m]}"
            )
            sys.exit(1)

    console.print(f"\n[bold]Running job: {args.job}[/]")
    console.print(f"  proteins: {', '.join(p.stem for p in input_paths)}")
    console.print(f"  results:  [dim]{result_dir}[/]")

    cli_params = {"msa_mode": args.msa_mode, "max_msa": args.max_msa,
                  "num_models": args.num_models, "num_seeds": args.num_seeds}
    overrides = {k: v for k, v in cli_params.items() if v != RUN_CONFIG[k]}

    if overrides:
        console.print(f"  [yellow]overrides[/]: {overrides}")

    if ablations:
        console.print(f"  [yellow]ablations[/]: {' → '.join(args.ablation)}")

    model_type = set_model_type(False, RUN_CONFIG["model_type"])
    download_alphafold_params(model_type)

    models = list(range(1, args.num_models + 1))
    seeds = list(range(RUN_CONFIG["random_seed"], RUN_CONFIG["random_seed"] + args.num_seeds))
    total_per_protein = len(models) * len(seeds)

    console.print()
    for input_path in input_paths:
        protein = input_path.stem
        logger.info("-" * 60)
        logger.info("PROTEIN: %s", protein)
        logger.info("-" * 60)
        # RNG independence: each ablation function creates a fresh
        # np.random.default_rng(seed) from the fixed CLI seed, so the
        # result is identical regardless of protein ordering or whether
        # this is a fresh run vs. a resume after partial completion.
        ablated_path = apply_ablations(input_path, ablations)
        queries, is_complex = get_queries(ablated_path)

        protein_result_dir = result_dir / protein
        _ensure_drive_dir(protein_result_dir, result_dir)

        # Track whether the MSA/config have already been saved (from a
        # previous run or from an earlier iteration in this run) so that
        # resumed runs don't re-generate MSA output unnecessarily.
        msa_saved = (protein_result_dir / f"{protein}.a3m").exists()

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

            for model_num, seed in product(models, seeds):
                if prediction_exists(protein_result_dir, protein, model_num, seed):
                    skipped += 1
                    console.print(f"  [blue]skipped[/]    {protein} model_{model_num} seed_{seed:03d}")
                    progress.advance(task)
                    continue

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
                            # RNG independence: random_seed is set explicitly
                            # per (model, seed) pair — no global RNG state.
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
                            skip_output=["plots"] if not msa_saved else ["plots", "msa"],
                        )

                        # Save results to Drive immediately via atomic copy
                        # (write to .tmp then rename, so prediction_exists()
                        # never sees a half-written file on resume).
                        for f in tmp_path.glob("*.pdb"):
                            _safe_copy(f, protein_result_dir / f.name)
                        for f in tmp_path.glob("*scores*.json"):
                            _safe_copy(f, protein_result_dir / f.name)

                        # Save MSA and config once per protein
                        if not msa_saved:
                            a3m = tmp_path / f"{protein}.a3m"
                            if a3m.exists():
                                _safe_copy(a3m, protein_result_dir / a3m.name)
                            config = tmp_path / "config.json"
                            if config.exists():
                                _safe_copy(config, protein_result_dir / config.name)
                            msa_saved = True

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
