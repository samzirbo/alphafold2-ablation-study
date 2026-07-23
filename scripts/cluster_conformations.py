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

Caching
-------
The all-to-all TM-score distance matrix -- the only expensive artifact with no
cache elsewhere -- is computed once and written to the output folder as
``<protein>_distance_matrix.npy`` alongside ``<protein>_manifest.csv`` (the
ordered ``(experiment, model, seed)`` bookkeeping) and ``<protein>_cache_meta.json``
(the inputs it was built from). Re-running the same command reuses that cache, so
changing the colouring, axes, labels or ``--n_clusters`` re-plots in seconds
rather than recomputing the matrix. Pass ``--force`` to recompute it; changing
the pooled experiments, ``--sample`` or ``--random_seed`` invalidates it
automatically.

The per-prediction scores against the two references are *not* recomputed here:
they are read from the shared ``plots/TM_Score/<experiment>/<protein>.csv`` cache
written by ``auto_plot`` / ``plot_tmscore`` (and computed into it on a miss).
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
from matplotlib.ticker import FuncFormatter, MaxNLocator
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
    from plot_tmscore import (
        calc_tm_score_folders,
        plot_tm_score,
        TM_ANNOTATION_FONTSIZES,
    )
except ImportError:
    from scripts.plot_tmscore import (
        calc_tm_score_folders,
        plot_tm_score,
        TM_ANNOTATION_FONTSIZES,
    )

console = Console()

# subfolders of a results root that are not experiments
EXCLUDED_DIRS = {"archive", "plots"}

# colour for dendrogram links that join two different clusters (above the cut)
_MIXED_LINK_COLOR = "#b0b0b0"

# one sequential colormap per cluster for the "shaded" visualization: every
# prediction in a cluster gets a distinct shade of the cluster's hue. Assigned
# to clusters in sorted-id order and cycled if there are more clusters than maps.
_SHADE_CMAPS = ["Blues", "Reds", "Greens", "Purples", "Oranges", "PuRd", "GnBu", "YlOrBr"]
# sample the colormap within this span; kept broad so shades are easy to tell
# apart, but clear of the near-white low end (invisible points) and near-black
# high end (hue lost against the black marker edge).
_SHADE_LO, _SHADE_HI = 0.30, 0.95

# line width for the dendrogram links + leaf stems
_LINE_WIDTH = 2.2


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


def shade_palette(
    Z: np.ndarray, labels: np.ndarray
) -> tuple[dict, dict]:
    """
    Per-prediction shade colours for the "shaded" visualization.

    Each cluster is given one hue (a sequential colormap); every prediction in
    the cluster gets a distinct shade of that hue. Shades are assigned along the
    dendrogram's left-to-right leaf order, so within a cluster the gradient
    follows the tree layout.

    Returns ``(shade_of, base_of)`` where ``shade_of`` maps a leaf index (the
    row/col of the distance matrix) to its unique hex shade, and ``base_of`` maps
    a cluster id to a single representative hex colour (for a compact legend).
    """
    # leaf order without drawing; indices are the observation (leaf) ids
    leaf_order = dendrogram(Z, no_plot=True)["leaves"]

    members_in_order: dict[int, list[int]] = {}
    for idx in leaf_order:
        members_in_order.setdefault(int(labels[idx]), []).append(idx)

    shade_of: dict[int, str] = {}
    base_of: dict[int, str] = {}
    for i, c in enumerate(sorted(members_in_order)):
        cmap = plt.get_cmap(_SHADE_CMAPS[i % len(_SHADE_CMAPS)])
        members = members_in_order[c]
        if len(members) == 1:
            positions = [(_SHADE_LO + _SHADE_HI) / 2.0]
        else:
            positions = np.linspace(_SHADE_LO, _SHADE_HI, len(members))
        for idx, p in zip(members, positions):
            shade_of[idx] = to_hex(cmap(p))
        base_of[c] = to_hex(cmap((_SHADE_LO + _SHADE_HI) / 2.0))
    return shade_of, base_of


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
    tm_axis: tuple[float, float] | None = None,
    link_leaf_colors: dict | None = None,
    label_colors: dict | None = None,
) -> None:
    """
    Draw the dendrogram, coloured per cluster to match the TM-score scatter.

    ``link_leaf_colors`` / ``label_colors`` optionally override the per-leaf
    colours used for the links and the tick labels respectively (keyed by leaf
    index). Both default to the per-cluster palette. Passing a per-cluster base
    colour for the links and a per-prediction shade for the labels yields the
    "shaded" variant where every prediction is a distinct shade of its cluster.
    """
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
    default_leaf_color = {i: palette[cluster_label(cluster_labels[i])] for i in range(n)}
    link_leaf_color = link_leaf_colors if link_leaf_colors is not None else default_leaf_color
    label_leaf_color = label_colors if label_colors is not None else default_leaf_color
    link_color = {}
    for i, (a, b) in enumerate(Z[:, :2].astype(int)):
        ca = link_color[a] if a >= n else link_leaf_color[a]
        cb = link_color[b] if b >= n else link_leaf_color[b]
        link_color[i + n] = ca if ca == cb else _MIXED_LINK_COLOR

    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    # scipy draws the links at the default line width; bump it via rc_context so
    # the whole tree is thicker (the overplotted leaf stems below match).
    with plt.rc_context({"lines.linewidth": _LINE_WIDTH}):
        rendered = dendrogram(
            Z,
            labels=leaf_labels if show_labels else None,
            no_labels=not show_labels,
            link_color_func=lambda k: link_color[k],
            ax=ax,
        )
    # tint the leaf tick labels to match their (cluster or per-prediction) colour
    if show_labels:
        for tick, leaf_idx in zip(ax.get_xticklabels(), rendered["leaves"]):
            tick.set_color(label_leaf_color[leaf_idx])

    # Colour each prediction's own vertical leaf stem with its shade. scipy paints
    # a whole link (U shape) one colour via link_color_func, so the per-prediction
    # shades can't come from there; instead overplot each leaf stem -- the arm
    # whose bottom sits at height 0 -- from 0 up to its first merge, in the leaf's
    # colour. Only done when per-leaf label_colors are given (the shaded variant);
    # otherwise the stems already match their cluster's link colour.
    if label_colors is not None:
        leaves = rendered["leaves"]
        for xs, ys in zip(rendered["icoord"], rendered["dcoord"]):
            # xs = [x_left, x_left, x_right, x_right]; ys = [y_left, ymerge, ymerge, y_right]
            for x_arm, y_bottom, y_top in ((xs[0], ys[0], ys[1]), (xs[3], ys[3], ys[2])):
                if y_bottom == 0.0:  # this arm descends to a leaf
                    pos = int(round((x_arm - 5.0) / 10.0))
                    if 0 <= pos < len(leaves):
                        ax.plot(
                            [x_arm, x_arm],
                            [0.0, y_top],
                            color=label_leaf_color[leaves[pos]],
                            linewidth=_LINE_WIDTH,
                            solid_capstyle="butt",
                            zorder=3,
                        )
    # if n_clusters > 1:
    #     ax.axhline(threshold, color="black", linestyle="--", linewidth=0.8)

    # The dendrogram plots merge heights in distance space (height = 1 - TM), so
    # bottom = height 0 = TM 1. Relabel the ticks as TM-score (1 - height) so the
    # axis reads as TM-score descending upward, without changing the geometry.
    # Fewer, larger ticks via MaxNLocator + a formatter that maps height -> TM.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda h, _pos: f"{1 - h:.2f}"))

    ax.set_ylabel("TM-score", fontsize=font_size + 5, fontweight="bold")
    ax.set_xlabel("prediction", fontsize=font_size + 5, fontweight="bold")
    ax.set_title(
        f"{experiment_name} | {protein} | {n_clusters}-cluster dendrogram",
        fontsize=font_size + 5,
        fontweight="bold",
    )
    ax.tick_params(axis="x", labelsize=font_size - 1, rotation=90)
    ax.tick_params(axis="y", labelsize=font_size + 4)

    # optionally fix the TM-score span shown. The axis is in height space
    # (height = 1 - TM), so TM range (tm_bottom, tm_top) maps to ylim
    # (1 - tm_bottom, 1 - tm_top). This may clip or squish the tree, which is
    # acceptable per the caller's request.
    if tm_axis is not None:
        tm_bottom, tm_top = tm_axis
        ax.set_ylim(1 - tm_bottom, 1 - tm_top)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def tmscore_cache_csv(folder: str, protein: str) -> Path:
    """
    Canonical location of the reference-score cache for one experiment folder,
    the CSV written by ``auto_plot`` / ``plot_tmscore``:
    ``<results_root>/plots/TM_Score/<experiment>/<protein>.csv``.

    ``folder`` is an experiment folder like ``results/depth_256``; the cache
    lives next to it under the sibling ``plots/TM_Score`` tree.
    """
    folder = Path(folder)
    return folder.parent / "plots" / "TM_Score" / folder.name / f"{protein}.csv"


def scores_vs_references(
    protein: str,
    experiment_folders: list[str],
    present_experiments: set[str],
    reference_folder: str,
    metadata_file: str,
) -> pd.DataFrame:
    """
    TM-score of every pooled prediction against the two references, one
    experiment at a time, concatenated into a single dataframe with a correct
    ``experiment`` column.

    Reuses the shared ``plots/TM_Score/<experiment>/<protein>.csv`` cache
    produced by ``auto_plot`` / ``plot_tmscore`` when it exists. On a miss the
    scores are computed with ``calc_tm_score_folders`` straight into that
    canonical cache location (the same thing ``auto_plot`` does), so nothing is
    duplicated and the next run -- clustering or plotting -- reuses it.
    """
    frames = []
    for folder in experiment_folders:
        experiment_name = Path(folder).name
        if experiment_name not in present_experiments:
            continue
        cache_csv = tmscore_cache_csv(folder, protein)
        if not cache_csv.exists():
            cache_csv.parent.mkdir(parents=True, exist_ok=True)
            calc_tm_score_folders(
                protein,
                reference_folder,
                os.path.join(folder, protein),
                metadata_file,
                f"{protein}.csv",
                str(cache_csv.parent),
            )
        df = pd.read_csv(cache_csv)
        # the cache's "experiment" column is the folder name by construction, but
        # set it explicitly so pooled rows stay attributable.
        df["experiment"] = experiment_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Distance-matrix cache
#
# The all-to-all TM-score distance matrix is the one genuinely expensive artifact
# with no cache elsewhere (reference scores are already cached under
# plots/TM_Score, see scores_vs_references). It is computed once and cached, so
# re-plotting with different colouring, axes, labels or --n_clusters reuses it
# and is effectively instant -- clustering, the dendrogram and the scatter are
# all cheap and re-derived on every run.
#
# The cache's canonical order is the pooled `origins` list; each prediction is
# identified by (experiment, model, seed). That ordered manifest is the "book
# keeping" that ties the distance-matrix rows, the dendrogram leaves and the
# scatter rows together, so cluster ids -- and therefore colours -- stay
# consistent across the dendrogram and the TM-score plot without caching the
# colours themselves.
# ---------------------------------------------------------------------------


def _cache_paths(out: str, protein: str) -> dict:
    """Paths of the cached distance-matrix artifacts for ``protein``."""
    return {
        "D": os.path.join(out, f"{protein}_distance_matrix.npy"),
        "manifest": os.path.join(out, f"{protein}_manifest.csv"),
        "meta": os.path.join(out, f"{protein}_cache_meta.json"),
    }


def _cache_meta(experiment_folders: list[str], sample: int | None, random_seed: int) -> dict:
    """Inputs that determine the cached matrix; used to auto-invalidate it."""
    return {
        "experiment_folders": list(experiment_folders),
        "sample": sample,
        "random_seed": random_seed,
    }


def load_matrix_cache(out: str, protein: str, meta: dict):
    """
    Load the cached ``(manifest, D)`` for ``protein`` if the artifacts exist and
    were built from the same inputs (``meta``). Returns ``None`` when there is no
    valid cache, so the caller recomputes.
    """
    paths = _cache_paths(out, protein)
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    with open(paths["meta"]) as fh:
        if json.load(fh) != meta:
            return None
    manifest = pd.read_csv(paths["manifest"])
    D = np.load(paths["D"])
    return manifest, D


def compute_matrix_cache(
    protein: str,
    experiment_folders: list[str],
    out: str,
    sample: int | None,
    rng: random.Random,
    meta: dict,
):
    """
    Pool predictions, compute the all-to-all distance matrix and cache it with
    its ordering manifest. Returns ``(manifest, D)`` in the canonical
    ``origins`` order, or ``None`` if fewer than two predictions were pooled.
    """
    # pool predictions across all requested experiments (optionally subsampled)
    origins = gather_predictions(protein, experiment_folders, sample=sample, rng=rng)
    if len(origins) < 2:
        return None

    files = [f for _, f in origins]
    present_experiments = {e for e, _ in origins}
    console.print(
        f"  pooled {len(files)} predictions from {len(present_experiments)} experiment(s)"
    )

    # canonical manifest: (experiment, model, seed) identifies each prediction;
    # `order` is its row/column in the distance matrix below.
    keys = [(exp, *_model_seed(f)) for exp, f in origins]
    manifest = pd.DataFrame(
        {
            "order": range(len(origins)),
            "experiment": [k[0] for k in keys],
            "model": [k[1] for k in keys],
            "seed": [k[2] for k in keys],
            "pdb_path": files,
        }
    )

    D = compute_distance_matrix(files)

    paths = _cache_paths(out, protein)
    np.save(paths["D"], D)
    manifest.to_csv(paths["manifest"], index=False)
    with open(paths["meta"], "w") as fh:
        json.dump(meta, fh, indent=2)

    return manifest, D


def cluster_and_plot(
    protein: str,
    run_name: str,
    out: str,
    manifest: pd.DataFrame,
    D: np.ndarray,
    scores: pd.DataFrame,
    n_clusters: int,
    font_size: int,
    tm_axis: tuple[float, float] | None = None,
) -> None:
    """
    The cheap, re-runnable half: cluster the cached distance matrix and draw the
    dendrogram + TM-score scatter. Everything here is fast and re-derived, so it
    can be re-run freely to change colouring, axes, labels or ``n_clusters``.
    """
    single_experiment = manifest["experiment"].nunique() == 1
    # canonical order comes from the manifest, matching the rows/cols of D
    keys = list(
        zip(
            manifest["experiment"],
            manifest["model"].astype(int),
            manifest["seed"].astype(int),
        )
    )

    # dendrogram clustering -> cut into n_clusters groups (cheap; re-derived)
    Z, labels = cluster_distance_matrix(D, n_clusters)

    leaf_labels = [
        f"m{m}s{s}" if single_experiment else f"{exp}:m{m}s{s}"
        for exp, m, s in keys
    ]

    plot_dendrogram(
        Z,
        leaf_labels,
        labels,
        n_clusters,
        protein,
        run_name,
        os.path.join(out, f"{protein}_dendrogram.png"),
        font_size=font_size,
        tm_axis=tm_axis,
    )

    # "shaded" variant: one hue per cluster, a distinct shade per prediction,
    # shared between this dendrogram and the scatter below so each leaf can be
    # matched to its point by colour. Links use the cluster's base hue; the leaf
    # labels use each prediction's unique shade.
    shade_of, base_of = shade_palette(Z, labels)
    link_leaf_colors = {i: base_of[int(labels[i])] for i in range(len(labels))}
    plot_dendrogram(
        Z,
        leaf_labels,
        labels,
        n_clusters,
        protein,
        run_name,
        os.path.join(out, f"{protein}_dendrogram_shaded.png"),
        font_size=font_size,
        tm_axis=tm_axis,
        link_leaf_colors=link_leaf_colors,
        label_colors=shade_of,
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

    # human-readable, label-indexed copy of the matrix + the cluster summary
    pd.DataFrame(
        D, index=leaf_labels, columns=leaf_labels
    ).to_csv(os.path.join(out, f"{protein}_distance_matrix_labeled.csv"))
    with open(os.path.join(out, f"{protein}_cluster_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # attach the cluster id of each prediction to the reference scores and plot
    # coloured by cluster.
    cluster_of = {key: int(c) for key, c in zip(keys, labels)}
    # the reference scores cover every prediction in each folder; keep only the
    # ones actually pooled/clustered (relevant when --sample is used).
    row_keys = list(
        zip(scores["experiment"], scores["model"].astype(int), scores["seed"].astype(int))
    )
    keep = [k in cluster_of for k in row_keys]
    df = scores[keep].copy()
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
        **TM_ANNOTATION_FONTSIZES,
    )

    # shaded scatter: each point gets its prediction's unique shade (matching the
    # shaded dendrogram). The legend shows one representative colour per cluster.
    key_to_index = {k: i for i, k in enumerate(keys)}
    kept_keys = [k for k in row_keys if k in cluster_of]
    df_shaded = df.copy()
    df_shaded["shade"] = [shade_of[key_to_index[k]] for k in kept_keys]
    shaded_csv = os.path.join(out, f"{protein}_shades.csv")
    df_shaded.to_csv(shaded_csv, index=False)

    plot_tm_score(
        data_file=shaded_csv,
        save_file_name=f"{protein}_shades.png",
        protein=protein,
        output_dir=out,
        experiment_name=f"{run_name} ",
        font_size=font_size,
        color_on="shade",
        literal_colors=True,
        legend_entries=[(cluster_label(c), base_of[c]) for c in sorted(base_of)],
        legend_title="cluster",
        **TM_ANNOTATION_FONTSIZES,
    )
    console.print(f"  [green]done[/] -> {out}")


def cluster_protein(
    protein: str,
    experiment_folders: list[str],
    data_dir: str,
    output_dir: str,
    run_name: str,
    n_clusters: int = 2,
    font_size: int = 8,
    sample: int | None = None,
    random_seed: int = 0,
    force: bool = False,
    tm_axis: tuple[float, float] | None = None,
) -> None:
    reference_folder = os.path.join(data_dir, protein, "references") + "/"
    metadata_file = os.path.join(data_dir, "metadata.json")

    out = os.path.join(output_dir, run_name)
    os.makedirs(out, exist_ok=True)

    console.print(f"[bold cyan]{run_name}[/] | [bold]{protein}[/]")

    # reuse the cached distance matrix unless the inputs changed or --force was
    # passed; otherwise pool the predictions and compute + cache it.
    meta = _cache_meta(experiment_folders, sample, random_seed)
    cache = None if force else load_matrix_cache(out, protein, meta)
    if cache is None:
        # per-protein RNG so a cache hit on one protein can't shift another
        # protein's --sample draw; a given (random_seed, protein) is reproducible.
        rng = random.Random(f"{random_seed}:{protein}")
        cache = compute_matrix_cache(
            protein, experiment_folders, out, sample, rng, meta,
        )
        if cache is None:
            console.print("  [yellow]skip[/] fewer than 2 predictions to cluster")
            return
    else:
        console.print("  [dim]reusing cached distance matrix[/]")

    manifest, D = cache
    if len(manifest) < n_clusters:
        console.print(
            f"  [yellow]skip[/] only {len(manifest)} predictions for {n_clusters} clusters"
        )
        return

    # reference scores (reuses the shared plots/TM_Score cache; cheap)
    scores = scores_vs_references(
        protein,
        experiment_folders,
        set(manifest["experiment"]),
        reference_folder,
        metadata_file,
    )

    cluster_and_plot(
        protein, run_name, out, manifest, D, scores, n_clusters, font_size,
        tm_axis=tm_axis,
    )


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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute the distance matrix even if a matching cache exists",
    )
    parser.add_argument(
        "--tm_axis",
        nargs=2,
        type=float,
        metavar=("TM_BOTTOM", "TM_TOP"),
        default=None,
        help="Fix the dendrogram TM-score axis, bottom then top, "
        "e.g. --tm_axis 1.0 0.5. May clip/squish the tree; default: auto",
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
                random_seed=args.random_seed,
                force=args.force,
                tm_axis=tuple(args.tm_axis) if args.tm_axis is not None else None,
            )
        except Exception as e:  # keep going across proteins
            console.print(f"  [red]failed[/] {protein}: {e}")


if __name__ == "__main__":
    main()
