"""Generate aligned PNGs and optional rocking GIFs for experiment predictions.

Predicted structures are aligned to the chain-specific metadata ``state_1``
reference before rendering. The CLI stays intentionally small; low-level PyMOL
settings live in ``src.utils.pymol_rendering``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from src.utils.pymol_rendering import PymolRenderer

console = Console(highlight=False)

ROOT = Path(__file__).resolve().parent.parent
PDB_PATTERN = "*_unrelaxed_*_alphafold2_model_*_seed_*.pdb"
EXCLUDED_EXPERIMENTS = {"archive", "plots"}
PROTEINS = [
    "PfMATE",
    "MCT1",
    "MurJ",
    "SERT",
    "ASCT2",
    "PTH1R",
    "STP10",
    "CGRPR",
    "LAT1",
    "ZnT8",
    "CCR5",
    "FZD7",
]

IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 1000
IMAGE_DPI = 200
ZOOM_BUFFER = 8.0
GIF_FRAMES = 40
GIF_DURATION = 0.20
Stats = dict[str, int]


@dataclass(frozen=True)
class RenderJob:
    experiment: str
    protein: str
    pdb_path: Path
    png_path: Path
    gif_path: Path
    reference_pdb: Path


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


def resolve_experiments(result_path: Path, experiments_requested: list[str] | None) -> list[Path]:
    """Return experiments to render, excluding archive/plot/hidden folders."""
    if not result_path.is_dir():
        raise FileNotFoundError(f"Result path not found: {result_path}")

    experiments = sorted(
        path
        for path in result_path.iterdir()
        if path.is_dir()
        and path.name not in EXCLUDED_EXPERIMENTS
        and not path.name.startswith(".")
    )

    if experiments_requested is None:
        return experiments

    resolved: list[Path] = []
    seen: set[Path] = set()
    for experiment in experiments_requested:
        matches = [path for path in experiments if path.name == experiment]
        if not matches:
            matches = [path for path in experiments if experiment in path.name]
        for path in matches:
            if path not in seen:
                resolved.append(path)
                seen.add(path)
    return resolved


def reference_pdb_for_protein(base_repo_path: Path, metadata: dict, protein: str) -> Path:
    """Resolve the chain-specific ``state_1`` reference used for predictions."""
    state = metadata[protein]["conformations"]["state_1"]
    name = f"{state['label']}_{state['pdb_id']}_{state['chain']}.pdb"
    return base_repo_path / "data" / protein / "references" / name


def discover_jobs(
    result_path: Path,
    output_path: Path,
    base_repo_path: Path,
    experiments: list[str] | None,
    proteins: list[str],
) -> tuple[list[RenderJob], list[str]]:
    """Discover prediction PDBs and their output paths."""
    metadata = load_metadata(base_repo_path)
    jobs: list[RenderJob] = []
    warnings: list[str] = []

    for experiment_dir in resolve_experiments(result_path, experiments):
        for protein in proteins:
            protein_dir = experiment_dir / protein
            if not protein_dir.is_dir():
                continue

            reference_pdb = reference_pdb_for_protein(base_repo_path, metadata, protein)
            if not reference_pdb.exists():
                warnings.append(f"{experiment_dir.name}/{protein}: missing reference {reference_pdb}")
                continue

            for pdb_path in sorted(protein_dir.glob(PDB_PATTERN), key=pdb_sort_key):
                out_dir = output_path / experiment_dir.name / protein
                jobs.append(
                    RenderJob(
                        experiment=experiment_dir.name,
                        protein=protein,
                        pdb_path=pdb_path,
                        png_path=out_dir / f"{pdb_path.stem}.png",
                        gif_path=out_dir / f"{pdb_path.stem}.gif",
                        reference_pdb=reference_pdb,
                    )
                )

    return jobs, warnings


def group_by(items, key_fn) -> dict[str, list]:
    groups: dict[str, list] = {}
    for item in items:
        groups.setdefault(key_fn(item), []).append(item)
    return groups


def new_stats() -> Stats:
    return {"png": 0, "gif": 0, "skipped": 0, "failed": 0}


def output_plan(job: RenderJob, *, gif: bool, force: bool) -> tuple[bool, bool]:
    make_png = force or not job.png_path.exists()
    make_gif = gif and (force or not job.gif_path.exists())
    return make_png, make_gif


def render_visualizations(
    jobs: list[RenderJob],
    *,
    gif: bool,
    gif_frames: int,
    force: bool,
) -> tuple[dict[str, int], list[str]]:
    """Render jobs grouped by experiment and protein for stable per-protein views."""
    stats = new_stats()
    failures: list[str] = []
    experiments = group_by(jobs, lambda job: job.experiment)

    with PymolRenderer(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        dpi=IMAGE_DPI,
        zoom_buffer=ZOOM_BUFFER,
    ) as renderer:
        for i, (experiment, experiment_jobs) in enumerate(experiments.items(), start=1):
            experiment_stats = new_stats()

            console.print(f"\n[bold cyan]Experiment {i}/{len(experiments)}:[/] {experiment}")
            progress = Progress(
                SpinnerColumn(),
                TextColumn("  {task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("proteins"),
                console=console,
                transient=True,
            )

            with progress:
                protein_groups = group_by(experiment_jobs, lambda job: job.protein)
                task = progress.add_task("proteins", total=len(protein_groups))

                for protein, protein_jobs in protein_groups.items():
                    progress.update(task, description=protein)
                    planned_jobs = []
                    for job in protein_jobs:
                        make_png, make_gif = output_plan(job, gif=gif, force=force)
                        if make_png or make_gif:
                            planned_jobs.append((job, make_png, make_gif))
                        else:
                            stats["skipped"] += 1
                            experiment_stats["skipped"] += 1

                    if not planned_jobs:
                        progress.advance(task)
                        continue

                    try:
                        # One fixed view per protein makes all predictions visually comparable.
                        renderer.prepare_fixed_view(
                            [job.pdb_path for job in protein_jobs],
                            reference_pdb=protein_jobs[0].reference_pdb,
                        )
                    except Exception as e:
                        for job, _, _ in planned_jobs:
                            stats["failed"] += 1
                            experiment_stats["failed"] += 1
                            failures.append(f"{job.experiment}/{job.protein}/{job.pdb_path.name}: {e}")
                        progress.advance(task)
                        continue

                    for job, make_png, make_gif in planned_jobs:
                        try:
                            renderer.load_structure(
                                job.pdb_path,
                                reference_pdb=job.reference_pdb,
                            )
                            if make_png:
                                renderer.save_png(job.png_path)
                                stats["png"] += 1
                                experiment_stats["png"] += 1
                            if make_gif:
                                renderer.save_gif(job.gif_path, frames=gif_frames, duration=GIF_DURATION)
                                stats["gif"] += 1
                                experiment_stats["gif"] += 1
                        except Exception as e:
                            stats["failed"] += 1
                            experiment_stats["failed"] += 1
                            failures.append(f"{job.experiment}/{job.protein}/{job.pdb_path.name}: {e}")

                    progress.advance(task)

            console.print(
                f"  [green]done[/] {experiment}: "
                f"PNG {experiment_stats['png']}, GIF {experiment_stats['gif']}, "
                f"skipped {experiment_stats['skipped']}, failed {experiment_stats['failed']}"
            )

    return stats, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--base_repo_path", type=Path, default=ROOT)
    parser.add_argument("--experiment", nargs="+", required=False)
    parser.add_argument("--protein", nargs="+", choices=PROTEINS, default=None)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--gif-frames", type=int, default=GIF_FRAMES)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = args.result_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    base_repo_path = args.base_repo_path.expanduser().resolve()
    proteins = args.protein if args.protein is not None else PROTEINS

    if args.experiment is None:
        console.print("[bold]Generating visualizations for all experiments[/]")
    else:
        console.print(f"[bold]Generating visualizations for experiments[/] {', '.join(args.experiment)}")
    console.print(f"  results: [dim]{result_path}[/]")
    console.print(f"  output:  [dim]{output_path}[/]")
    console.print(f"  mode:    PNG{' + GIF' if args.gif else ''}")
    console.print("  color:   [cyan]sequence-id[/]")

    try:
        jobs, warnings = discover_jobs(
            result_path,
            output_path,
            base_repo_path,
            args.experiment,
            proteins,
        )
    except Exception as e:
        console.print(f"[red bold]FAILED[/] {e}")
        sys.exit(1)

    for warning in warnings:
        console.print(f"  [yellow]warning[/] {warning}")
    if not jobs:
        console.print("  [yellow]warning[/] No matching PDB files found.")
        return

    stats, failures = render_visualizations(
        jobs,
        gif=args.gif,
        gif_frames=args.gif_frames,
        force=args.force,
    )

    console.print(
        f"\n[bold green]Visualization complete.[/] "
        f"PNG: {stats['png']}, GIF: {stats['gif']}, "
        f"skipped: {stats['skipped']}, failed: {stats['failed']}\n"
    )

    if failures:
        console.print("[red bold]Failures:[/]")
        for failure in failures:
            console.print(f"  [red]-[/] {failure}")
        sys.exit(1)


if __name__ == "__main__":
    main()
