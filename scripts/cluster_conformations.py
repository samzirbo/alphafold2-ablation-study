"""
All-to-all TM-score clustering of AlphaFold predictions.

For each protein, the predictions from one or more experiment folders are pooled
and this:

1. computes the all-to-all TM-score distance matrix over the pooled predictions
   (distance = 1 - TM),
2. runs hierarchical (UPGMA / average-linkage) dendrogram clustering and cuts
   the tree into ``--n_clusters`` groups (2 by default),
3. plots the dendrogram,
4. scores every prediction against the two experimental references and plots the
   inward/outward (or inactive/active) TM-score scatter using the existing
   ``plot_tmscore`` infrastructure, colouring the points by the cluster the
   dendrogram assigned.

Run from the repository root::

    # a single experiment
    python ./scripts/cluster_conformations.py \\
        --experiment_folders ./results/depth_128 --output_dir ./analysis/clustering

    # pool several experiments together
    python ./scripts/cluster_conformations.py \\
        --experiment_folders ./results/depth_128 ./results/row_mask_128 \\
        --output_dir ./analysis/clustering

    # pool every experiment inside a results directory
    python ./scripts/cluster_conformations.py \\
        --all_experiments_in ./results --output_dir ./analysis/clustering

    # subsample 10 of 25 predictions per experiment to bound the O(N^2) cost
    python ./scripts/cluster_conformations.py \\
        --all_experiments_in ./results --sample 10 --output_dir ./analysis/clustering

Omit ``--protein`` to process every protein found in the experiment folder(s).
"""

import argparse
import json
import os
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from tmtools import tm_align
from tmtools.io import get_residue_data, get_structure

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

# Support both ``python ./scripts/cluster_conformations.py`` (scripts/ on path)
# and ``python -m scripts.cluster_conformations`` (repo root on path).
try:
    from plot_tmscore import calc_tm_score_folders, plot_tm_score
except ImportError:
    from scripts.plot_tmscore import calc_tm_score_folders, plot_tm_score

console = Console()

# subfolders of a results root that are not experiments
EXCLUDED_DIRS = {"archive", "plots"}

# colour for dendrogram links that join two different clusters (above the cut)
_MIXED_LINK_COLOR = "#b0b0b0"


def cluster_label(c) -> str:
    """Label for a cluster, used for both the CSV/legend text and palette keys."""
    return f"Cluster {int(c)}"


def cluster_palette(cluster_labels) -> dict:
    """
    Colour per cluster label, built identically to how ``plot_tm_score`` colours a
    *categorical* ``cluster`` column: the Okabe-Ito colormap assigned by the
    sorted position of the label (``np.sort`` of the categories, then
    ``okabe_ito(i % N)``). Reproducing that exact construction here guarantees the
    dendrogram and the TM-score scatter share one colour per cluster, regardless
    of any environment default colormap.

    Returns a ``{label_string: hex_colour}`` dict.
    """
    okabe = plt.cm.okabe_ito
    categories = np.sort([cluster_label(c) for c in {int(x) for x in cluster_labels}])
    return {cat: to_hex(okabe(i % okabe.N)) for i, cat in enumerate(categories)}


def _model_seed(target_file: str) -> tuple[int, int]:
    """Parse (model, seed) from a ``..._model_<m>_seed_<s>.pdb`` filename."""
    stem = Path(target_file).stem
    parts = stem.split("_")
    return int(parts[-3]), int(parts[-1])


def get_target_files(protein: str, target_folder: str) -> list[str]:
    """Return the sorted list of prediction .pdb files for ``protein``."""
    target_folder = target_folder.rstrip("/") + "/"
    files = sorted(
        str(f)
        for f in Path(target_folder).glob(
            f"{protein}_unrelaxed_*_alphafold2_model_*_seed_*.pdb"
        )
    )
    if len(files) != 25:
        warnings.warn(
            f"There are {len(files)} models in {target_folder} instead of 25"
        )
    return files


def gather_predictions(
    protein: str,
    experiment_folders: list[str],
    sample: int | None = None,
    rng: random.Random | None = None,
) -> list[tuple[str, str]]:
    """
    Pool the predictions of ``protein`` across ``experiment_folders``.

    Returns a list of ``(experiment_name, pdb_path)`` in a stable order.
    Experiment folders that don't contain the protein are silently skipped.

    If ``sample`` is given, at most ``sample`` predictions are drawn (without
    replacement) from each experiment, using ``rng`` for reproducibility. This
    keeps the all-to-all distance matrix -- which grows with the square of the
    pooled prediction count -- tractable.
    """
    origins: list[tuple[str, str]] = []
    for folder in experiment_folders:
        experiment_name = Path(folder).name
        target_folder = os.path.join(folder, protein)
        if not Path(target_folder).is_dir():
            continue
        files = get_target_files(protein, target_folder)
        if sample is not None and len(files) > sample:
            chooser = rng if rng is not None else random
            files = sorted(chooser.sample(files, sample))
        for f in files:
            origins.append((experiment_name, f))
    return origins


def compute_distance_matrix(target_files: list[str]) -> np.ndarray:
    """
    All-to-all TM-score distance matrix, distance = 1 - TM.

    Structures are parsed once and reused across all pairs. The TM-score is
    ``tm_align(...).tm_norm_chain1`` -- the same quantity used everywhere else in
    this repo (see ``plot_tmscore.__calc_tm_score``). Because every prediction
    is the same sequence/length the score is symmetric, so we fill the upper
    triangle and mirror it.
    """
    n = len(target_files)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("  loading structures"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    )
    parsed = []
    with progress:
        task = progress.add_task("loading", total=n)
        for f in target_files:
            structure = get_structure(f)
            coords, seq = get_residue_data(next(structure.get_chains()))
            parsed.append((coords, seq))
            progress.advance(task)

    D = np.zeros((n, n))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    progress = Progress(
        SpinnerColumn(),
        TextColumn("  all-to-all TM-score"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    )
    with progress:
        task = progress.add_task("tm", total=len(pairs))
        for i, j in pairs:
            coords_i, seq_i = parsed[i]
            coords_j, seq_j = parsed[j]
            tm = tm_align(coords_i, coords_j, seq_i, seq_j).tm_norm_chain1
            D[i, j] = D[j, i] = 1.0 - tm
            progress.advance(task)

    return D


def cluster_distance_matrix(
    D: np.ndarray, n_clusters: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Average-linkage hierarchical clustering of a precomputed distance matrix.

    Returns ``(linkage_matrix, labels)`` where ``labels`` is a length-n array of
    cluster ids in ``1..n_clusters``.
    """
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return Z, labels


def cluster_summary(D: np.ndarray, labels: np.ndarray) -> dict:
    """
    Per-cluster medoid + inter-cluster TM-score, derived from the distance
    matrix. The medoid is the prediction with the smallest total distance to the
    rest of its cluster (a real structure, not a synthetic average).
    """
    summary = {"clusters": {}, "between": {}}
    cluster_ids = sorted(set(labels.tolist()))

    medoids = {}
    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        # total within-cluster distance for each member; argmin == medoid
        within = D[np.ix_(idx, idx)].sum(axis=1)
        medoid = int(idx[int(np.argmin(within))])
        medoids[c] = medoid
        # mean pairwise TM inside the cluster (1 - mean distance over off-diagonal)
        if len(idx) > 1:
            off = D[np.ix_(idx, idx)][~np.eye(len(idx), dtype=bool)]
            mean_tm = 1.0 - float(off.mean())
        else:
            mean_tm = float("nan")
        summary["clusters"][c] = {
            "size": int(len(idx)),
            "medoid_index": medoid,
            "mean_internal_tm": mean_tm,
        }

    for a in range(len(cluster_ids)):
        for b in range(a + 1, len(cluster_ids)):
            ca, cb = cluster_ids[a], cluster_ids[b]
            idx_a = np.where(labels == ca)[0]
            idx_b = np.where(labels == cb)[0]
            medoid_tm = 1.0 - float(D[medoids[ca], medoids[cb]])
            mean_tm = 1.0 - float(D[np.ix_(idx_a, idx_b)].mean())
            summary["between"][f"{ca}-{cb}"] = {
                "medoid_tm": medoid_tm,
                "mean_tm": mean_tm,
            }

    return summary


def plot_dendrogram(
    Z: np.ndarray,
    leaf_labels: list[str],
    cluster_labels: np.ndarray,
    n_clusters: int,
    protein: str,
    experiment_name: str,
    save_path: str,
    font_size: int = 8,
) -> None:
    """Draw the dendrogram, coloured per cluster to match the TM-score scatter."""
    # height at which the tree has exactly n_clusters branches: midpoint between
    # the (n_clusters-1)-th and n_clusters-th last merges.
    if n_clusters < len(Z) + 1:
        upper = Z[-(n_clusters - 1), 2] if n_clusters > 1 else Z[-1, 2] * 1.1
        lower = Z[-n_clusters, 2]
        threshold = (upper + lower) / 2.0
    else:
        threshold = 0.0

    # keep the figure legible when many predictions are pooled together
    n = len(leaf_labels)
    show_labels = n <= 40

    # colour each leaf by its cluster, then propagate up: a link is painted with
    # the cluster colour when its whole subtree is one cluster, else grey. This
    # ties link colours to the fcluster ids (same ids the scatter colours by).
    palette = cluster_palette(cluster_labels)
    leaf_color = {i: palette[cluster_label(cluster_labels[i])] for i in range(n)}
    link_color = {}
    for i, (a, b) in enumerate(Z[:, :2].astype(int)):
        ca = link_color[a] if a >= n else leaf_color[a]
        cb = link_color[b] if b >= n else leaf_color[b]
        link_color[i + n] = ca if ca == cb else _MIXED_LINK_COLOR

    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    rendered = dendrogram(
        Z,
        labels=leaf_labels if show_labels else None,
        no_labels=not show_labels,
        link_color_func=lambda k: link_color[k],
        ax=ax,
    )
    # tint the leaf tick labels to match their cluster colour
    if show_labels:
        for tick, leaf_idx in zip(ax.get_xticklabels(), rendered["leaves"]):
            tick.set_color(leaf_color[leaf_idx])
    # if n_clusters > 1:
    #     ax.axhline(threshold, color="black", linestyle="--", linewidth=0.8)

    ax.set_ylabel("1 - TM-score", fontsize=font_size + 1, fontweight="bold")
    ax.set_xlabel("prediction", fontsize=font_size + 1, fontweight="bold")
    ax.set_title(
        f"{experiment_name} | {protein} | {n_clusters}-cluster dendrogram",
        fontsize=font_size + 3,
        fontweight="bold",
    )
    ax.tick_params(axis="x", labelsize=font_size - 1, rotation=90)
    ax.tick_params(axis="y", labelsize=font_size)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def scores_vs_references(
    protein: str,
    experiment_folders: list[str],
    present_experiments: set[str],
    reference_folder: str,
    metadata_file: str,
    out: str,
) -> pd.DataFrame:
    """
    TM-score of every pooled prediction against the two references, one
    experiment at a time (reusing ``calc_tm_score_folders``), concatenated into a
    single dataframe with a correct ``experiment`` column.
    """
    frames = []
    for folder in experiment_folders:
        experiment_name = Path(folder).name
        if experiment_name not in present_experiments:
            continue
        target_folder = os.path.join(folder, protein)
        tmp_csv = calc_tm_score_folders(
            protein,
            reference_folder,
            target_folder,
            metadata_file,
            f"__tmp_{experiment_name}_{protein}.csv",
            out,
        )
        df = pd.read_csv(tmp_csv)
        # calc_tm_score_folders derives "experiment" from output_dir; overwrite it
        # with the true source experiment so pooled rows stay distinguishable.
        df["experiment"] = experiment_name
        frames.append(df)
        os.remove(tmp_csv)
    return pd.concat(frames, ignore_index=True)


def cluster_protein(
    protein: str,
    experiment_folders: list[str],
    data_dir: str,
    output_dir: str,
    run_name: str,
    n_clusters: int = 2,
    font_size: int = 8,
    sample: int | None = None,
    rng: random.Random | None = None,
) -> None:
    reference_folder = os.path.join(data_dir, protein, "references") + "/"
    metadata_file = os.path.join(data_dir, "metadata.json")

    out = os.path.join(output_dir, run_name)
    os.makedirs(out, exist_ok=True)

    console.print(f"[bold cyan]{run_name}[/] | [bold]{protein}[/]")

    # pool predictions across all requested experiments (optionally subsampled)
    origins = gather_predictions(protein, experiment_folders, sample=sample, rng=rng)
    if len(origins) < n_clusters:
        console.print(f"  [yellow]skip[/] only {len(origins)} predictions")
        return

    experiments = [e for e, _ in origins]
    files = [f for _, f in origins]
    present_experiments = set(experiments)
    single_experiment = len(present_experiments) == 1
    console.print(
        f"  pooled {len(files)} predictions from {len(present_experiments)} experiment(s)"
    )

    # 1. all-to-all distance matrix over the pooled predictions
    D = compute_distance_matrix(files)

    # 2. dendrogram clustering -> cut into n_clusters groups
    Z, labels = cluster_distance_matrix(D, n_clusters)

    # (experiment, model, seed) identifies each pooled prediction uniquely
    keys = [(exp, *_model_seed(f)) for exp, f in origins]
    leaf_labels = [
        f"m{m}s{s}" if single_experiment else f"{exp}:m{m}s{s}"
        for exp, m, s in keys
    ]

    # 3. dendrogram plot
    plot_dendrogram(
        Z,
        leaf_labels,
        labels,
        n_clusters,
        protein,
        run_name,
        os.path.join(out, f"{protein}_dendrogram.png"),
        font_size=font_size,
    )

    # cluster diagnostics (medoids + inter-cluster TM)
    summary = cluster_summary(D, labels)
    for c, info in summary["clusters"].items():
        console.print(
            f"  cluster {c}: n={info['size']}, medoid={leaf_labels[info['medoid_index']]}, "
            f"mean internal TM={info['mean_internal_tm']:.3f}"
        )
    for pair, info in summary["between"].items():
        console.print(
            f"  clusters {pair}: medoid-medoid TM={info['medoid_tm']:.3f}, "
            f"mean inter-cluster TM={info['mean_tm']:.3f}"
        )

    # persist distance matrix + summary for downstream use
    pd.DataFrame(
        D, index=leaf_labels, columns=leaf_labels
    ).to_csv(os.path.join(out, f"{protein}_distance_matrix.csv"))
    with open(os.path.join(out, f"{protein}_cluster_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # 4. score against the two references, attach the cluster id of each
    #    prediction, and plot coloured by cluster.
    df = scores_vs_references(
        protein, experiment_folders, present_experiments,
        reference_folder, metadata_file, out,
    )

    cluster_of = {key: int(c) for key, c in zip(keys, labels)}
    # calc_tm_score_folders scores every prediction in each folder; keep only the
    # ones that were actually pooled/clustered (relevant when --sample is used).
    row_keys = list(zip(df["experiment"], df["model"].astype(int), df["seed"].astype(int)))
    keep = [k in cluster_of for k in row_keys]
    df = df[keep].copy()
    # store the string label ("Cluster 1", ...) so plot_tm_score treats it as a
    # categorical column (discrete Okabe-Ito colours + legend) and so the label
    # survives the CSV round-trip without being re-parsed as an integer.
    df["cluster"] = [cluster_label(cluster_of[k]) for k in row_keys if k in cluster_of]
    csv_path = os.path.join(out, f"{protein}.csv")
    df.to_csv(csv_path, index=False)

    plot_tm_score(
        data_file=csv_path,
        save_file_name=f"{protein}_clusters.png",
        protein=protein,
        output_dir=out,
        experiment_name=f"{run_name} (clustered)",
        font_size=font_size,
        color_on="cluster",
    )
    console.print(f"  [green]done[/] -> {out}")


def discover_proteins(experiment_folders: list[str], data_dir: str) -> list[str]:
    """Proteins (known in metadata) that appear in any of the experiment folders."""
    metadata = json.load(open(os.path.join(data_dir, "metadata.json")))
    known = set(metadata.keys())
    found = set()
    for folder in experiment_folders:
        for d in Path(folder).iterdir():
            if d.is_dir() and d.name in known:
                found.add(d.name)
    return sorted(found)


def resolve_run_name(experiment_folders: list[str], run_name: str | None) -> str:
    """Name for the pooled run's output subfolder / plot titles."""
    if run_name is not None:
        return run_name
    names = [Path(f).name for f in experiment_folders]
    if len(names) == 1:
        return names[0]
    if len(names) <= 3:
        return "+".join(names)
    return f"pooled_{len(names)}_experiments"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--experiment_folders",
        nargs="+",
        help="One or more experiment folders whose predictions are pooled together",
    )
    source.add_argument(
        "--all_experiments_in",
        help="Pool every experiment subfolder found in this results directory",
    )
    parser.add_argument(
        "--protein",
        default=None,
        help="Single protein to process; if omitted, all proteins found",
    )
    parser.add_argument("--data_dir", default="data", help="Directory with references + metadata.json")
    parser.add_argument("--output_dir", default="analysis/clustering")
    parser.add_argument(
        "--run_name",
        default=None,
        help="Name of the pooled run (output subfolder + plot titles); auto-derived if omitted",
    )
    parser.add_argument("--n_clusters", type=int, default=2)
    parser.add_argument("--font_size", type=int, default=8)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample at most this many predictions per experiment "
        "(e.g. 10 of 25) to keep the all-to-all matrix tractable; default: all",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Seed for the --sample random draw (for reproducibility)",
    )
    args = parser.parse_args()

    if args.all_experiments_in is not None:
        root = Path(args.all_experiments_in)
        experiment_folders = sorted(
            str(d)
            for d in root.iterdir()
            if d.is_dir() and d.name not in EXCLUDED_DIRS
        )
        if not experiment_folders:
            console.print(f"[yellow]No experiment folders found in {root}[/]")
            return
    else:
        experiment_folders = args.experiment_folders

    run_name = resolve_run_name(experiment_folders, args.run_name)

    if args.protein is not None:
        proteins = [args.protein]
    else:
        proteins = discover_proteins(experiment_folders, args.data_dir)
        if not proteins:
            console.print("[yellow]No protein folders found in the given experiment(s)[/]")
            return

    sample_note = f", sampling {args.sample}/experiment" if args.sample else ""
    console.print(
        f"[bold]Clustering[/] {len(proteins)} protein(s) across "
        f"{len(experiment_folders)} experiment(s) as [cyan]{run_name}[/]{sample_note}"
    )

    # one RNG for the whole run so a given --random_seed is fully reproducible
    rng = random.Random(args.random_seed)

    for protein in proteins:
        try:
            cluster_protein(
                protein,
                experiment_folders,
                args.data_dir,
                args.output_dir,
                run_name,
                n_clusters=args.n_clusters,
                font_size=args.font_size,
                sample=args.sample,
                rng=rng,
            )
        except Exception as e:  # keep going across proteins
            console.print(f"  [red]failed[/] {protein}: {e}")


if __name__ == "__main__":
    main()
