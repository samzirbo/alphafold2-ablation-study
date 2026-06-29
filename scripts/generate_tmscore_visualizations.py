"""Generate per-residue TM-score colored PNGs for one protein.

Aligns each prediction to the chosen reference(s), colors by per-residue TM
contribution, and writes PNGs (and optional rocking GIFs).

Output naming:

``tmscore_{reference_slug}_{prediction_stem}.png``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from src.utils.per_residue_tm_score import compute_per_residue_tm_scores
from src.utils.pymol_tmscore_rendering import GIF_DURATION_DEFAULT, GIF_FRAMES_DEFAULT, TmScoreRenderer
from src.utils.visualization_paths import (
    PDB_PATTERN,
    load_metadata,
    pdb_sort_key,
    reference_pdb_for_state,
    reference_slug,
)

console = Console(highlight=False)

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--base_repo_path", type=Path, default=ROOT)
    parser.add_argument("--experiment", required=True, help="Experiment folder name under result_path.")
    parser.add_argument("--protein", required=True, help="Protein name (e.g. LAT1).")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--gif-frames", type=int, default=GIF_FRAMES_DEFAULT)
    parser.add_argument(
        "--reference-state",
        choices=["state_1", "state_2", "both"],
        default="both",
        help="Which metadata reference conformation(s) to score against.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = args.result_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    base_repo_path = args.base_repo_path.expanduser().resolve()
    experiment = args.experiment
    protein = args.protein

    experiment_dir = result_path / experiment
    protein_dir = experiment_dir / protein
    if not experiment_dir.is_dir():
        console.print(f"[red bold]FAILED[/] Experiment not found: {experiment_dir}")
        sys.exit(1)
    if not protein_dir.is_dir():
        console.print(f"[red bold]FAILED[/] Protein directory not found: {protein_dir}")
        sys.exit(1)

    metadata = load_metadata(base_repo_path)
    if protein not in metadata:
        console.print(f"[red bold]FAILED[/] Unknown protein in metadata.json: {protein}")
        sys.exit(1)

    reference_states = ("state_1", "state_2") if args.reference_state == "both" else (args.reference_state,)
    state_refs: list[tuple[str, Path, str]] = []
    for state_key in reference_states:
        ref_pdb = reference_pdb_for_state(base_repo_path, metadata, protein, state_key)
        if not ref_pdb.exists():
            console.print(f"[red bold]FAILED[/] Missing reference: {ref_pdb}")
            sys.exit(1)
        state_refs.append((state_key, ref_pdb, reference_slug(metadata, protein, state_key)))

    pdb_paths = sorted(protein_dir.glob(PDB_PATTERN), key=pdb_sort_key)
    if not pdb_paths:
        console.print(f"[red bold]FAILED[/] No prediction PDBs found in {protein_dir}")
        sys.exit(1)

    console.print(f"[bold]Generating TM-score visualizations[/]")
    console.print(f"  experiment: {experiment}")
    console.print(f"  protein:    {protein}")
    console.print(f"  results:    [dim]{result_path}[/]")
    console.print(f"  output:     [dim]{output_path}[/]")
    console.print(f"  mode:       TM-score PNG{' + TM-score GIF' if args.gif else ''}")

    out_dir = output_path / experiment / protein
    failures: list[str] = []
    png_count = 0
    gif_count = 0

    with TmScoreRenderer() as renderer:
        for state_key, ref_pdb, slug in state_refs:
            console.print(f"\n[bold cyan]Reference:[/] {slug}")
            try:
                renderer.prepare_fixed_view(ref_pdb)
            except Exception as exc:
                console.print(f"[red bold]FAILED[/] Could not prepare view for {slug}: {exc}")
                sys.exit(1)

            for pdb_path in pdb_paths:
                stem = f"tmscore_{slug}_{pdb_path.stem}"
                png_path = out_dir / f"{stem}.png"
                gif_path = out_dir / f"{stem}.gif"
                try:
                    tm_scores = compute_per_residue_tm_scores(ref_pdb, pdb_path)
                    console.print(
                        f"  [cyan]tm-score[/] {pdb_path.name}: {tm_scores.tm_score:.3f}"
                    )
                    renderer.load_tmscore_structure(pdb_path, ref_pdb, tm_scores)
                    title = f"{experiment} | TM={tm_scores.tm_score:.3f}"
                    renderer.save_tmscore_png(png_path, title=title)
                    png_count += 1
                    if args.gif:
                        renderer.save_tmscore_gif(
                            gif_path,
                            frames=args.gif_frames,
                            duration=GIF_DURATION_DEFAULT,
                        )
                        gif_count += 1
                    console.print(f"  [green]ok[/] {pdb_path.name}")
                except Exception as exc:
                    failures.append(f"{slug}/{pdb_path.name}: {exc}")
                    console.print(f"  [red]failed[/] {pdb_path.name}: {exc}")

    console.print(
        f"\n[bold green]TM-score visualization complete.[/] "
        f"PNG: {png_count}, GIF: {gif_count}, failed: {len(failures)}\n"
    )
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
