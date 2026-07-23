"""
Side-by-side TM-score + PCA plot for one experiment / protein.

Draws the TM-score scatter (similarity to the two experimental conformations) on
the left and the PCA of the predictions' Cα coordinates on the right, sharing:

* **shape = AlphaFold model.** Each of the (up to) five models gets one fixed
  marker, so model 1 is the same shape in both panels -- and across every plot
  this script makes -- letting a point be matched between the two panels by its
  shape. A single combined legend maps shape -> model.
* **colour = TM-score to one chosen reference conformation** ("the gradient
  reference"). The PCA superimposes every prediction onto that reference and the
  points are shaded by their TM-score to it; the TM-score panel is shaded by the
  *same* value with the *same* colormap, so one shared colorbar describes both.
  Which reference drives the gradient is named in the title.

The gradient's numeric span is fixed with ``--gradient_range VMIN VMAX`` so that
several plots (e.g. one per experiment) can be produced with an identical
colour axis and compared directly.

Run from the repository root, e.g.::

    uv run ./scripts/plot_tm_pca.py \\
        --protein ASCT2 \\
        --reference_folder ./data/ASCT2/references \\
        --target_folder ./results/depth_256/ASCT2 \\
        --metadata_path ./data/metadata.json \\
        --experiment_name depth_256 \\
        --gradient_reference state_1 \\
        --gradient_range 0.5 1.0 \\
        --output_dir ./analysis/tm_pca
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

# Support both ``python ./scripts/plot_tm_pca.py`` (scripts/ on path) and
# ``python -m scripts.plot_tm_pca`` (repo root on path), matching the other
# scripts in this folder.
try:
    from plot_tmscore import (
        get_reference_files,
        get_axis_lower_limit,
        __calc_tm_score,
    )
except ImportError:
    from scripts.plot_tmscore import (
        get_reference_files,
        get_axis_lower_limit,
        __calc_tm_score,
    )

console = Console()

# One fixed marker per AlphaFold model, so a given model is always the same
# shape -- in both panels and across every plot this script produces. Models are
# 1..5; the fallback list covers anything unexpected without collisions up to 10.
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def model_marker(model: int) -> str:
    """Deterministic marker for an AlphaFold model number (1-indexed)."""
    return _MARKERS[(int(model) - 1) % len(_MARKERS)]


def _model_seed(target_file: str) -> tuple[int, int]:
    """Parse ``(model, seed)`` from a ``..._model_<m>_seed_<s>.pdb`` filename."""
    parts = Path(target_file).stem.split("_")
    return int(parts[-3]), int(parts[-1])


def compute_tm_and_pca(
    protein: str,
    reference_folder: str,
    target_folder: str,
    metadata_file: str,
    gradient_ref_index: int,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    For every prediction of ``protein`` compute its TM-score to both experimental
    references and its 2-D PCA projection.

    The PCA is built from the Cα coordinates after superimposing each prediction
    onto the *gradient reference* (``reference_files[gradient_ref_index]``), so
    PC space and the colour gradient share that reference.

    Returns ``(df, explained_variance_ratio, reference_labels)`` where ``df`` has
    one row per prediction with columns ``model, seed, tm_state_1, tm_state_2,
    pc1, pc2`` and ``reference_labels`` is ``[state_1_label, state_2_label]``
    parsed from the reference filenames (e.g. ``["IF", "OF"]``).
    """
    reference_folder = reference_folder.rstrip("/") + "/"
    target_folder = target_folder.rstrip("/") + "/"

    reference_files = get_reference_files(protein, reference_folder, metadata_file)
    if len(reference_files) == 0:
        raise ValueError("No reference structures found")
    gradient_ref_file = reference_files[gradient_ref_index]

    # Reference labels for titles/axes: filenames look like ``IF_6RVX_A.pdb``.
    reference_labels = [Path(f).name.split("_")[0] for f in reference_files]

    target_files = sorted(
        str(f)
        for f in Path(target_folder).glob(
            f"{protein}_unrelaxed_*_alphafold2_model_*_seed_*.pdb"
        )
    )
    if len(target_files) != 25:
        warnings.warn(f"There are {len(target_files)} models instead of 25")
    if len(target_files) < 3:
        raise ValueError("Not enough structures for PCA")

    parser = PDBParser(QUIET=True)
    grad_structure = parser.get_structure("gradient_ref", gradient_ref_file)
    grad_ca = {
        residue.id[1]: residue["CA"]
        for residue in grad_structure.get_residues()
        if "CA" in residue
    }

    progress = Progress(
        SpinnerColumn(),
        TextColumn(f"  {protein} TM + PCA"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    )

    coords: list[np.ndarray] = []
    rows: list[dict] = []
    with progress:
        task = progress.add_task(protein, total=len(target_files))
        for target_file in target_files:
            structure = parser.get_structure("model", target_file)
            model_ca = {
                residue.id[1]: residue["CA"]
                for residue in structure.get_residues()
                if "CA" in residue
            }

            # superimpose onto the gradient reference over the shared residues,
            # then keep the aligned Cα coordinates as the PCA feature vector.
            common_res = sorted(set(grad_ca.keys()) & set(model_ca.keys()))
            sup = Superimposer()
            sup.set_atoms(
                [grad_ca[r] for r in common_res],
                [model_ca[r] for r in common_res],
            )
            sup.apply(structure.get_atoms())
            coords.append(np.array([model_ca[r].coord for r in common_res]).flatten())

            model, seed = _model_seed(target_file)
            tm_state_1, _, _ = __calc_tm_score(reference_files[0], target_file)
            tm_state_2, _, _ = __calc_tm_score(reference_files[1], target_file)
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "tm_state_1": tm_state_1,
                    "tm_state_2": tm_state_2,
                }
            )
            progress.advance(task)

    pca = PCA(n_components=2)
    proj = pca.fit_transform(np.array(coords))

    df = pd.DataFrame(rows)
    df["pc1"] = proj[:, 0]
    df["pc2"] = proj[:, 1]
    return df, pca.explained_variance_ratio_, reference_labels


def plot_tm_pca(
    protein: str,
    reference_folder: str,
    target_folder: str,
    metadata_file: str,
    experiment_name: str,
    gradient_reference: str = "state_1",
    gradient_range: tuple[float, float] | None = None,
    save_file_name: str | None = None,
    output_dir: str = ".",
    font_size: int = 12,
    limit_axis: bool = True,
    plot_guidelines: bool = True,
    cmap_name: str = "plasma",
    color_by_tm: bool = True,
) -> str:
    """
    Draw the combined TM-score (left) + PCA (right) figure and save it.

    :param gradient_reference: which conformation drives the colour gradient,
        ``"state_1"`` or ``"state_2"`` (as named in ``metadata.json``).
    :param gradient_range: ``(vmin, vmax)`` fixing the colour axis; pass the same
        value for several plots so their gradients line up. Defaults to this
        plot's own TM-score min/max.
    :param color_by_tm: if ``False``, drop the TM-score colour shading entirely --
        every point gets one neutral fill and no colorbar is drawn (shape still
        encodes the model). ``gradient_range`` / ``gradient_reference`` are then
        unused.
    :returns: the path the figure was written to.
    """
    gradient_ref_index = {"state_1": 0, "state_2": 1}[gradient_reference]

    df, explained, reference_labels = compute_tm_and_pca(
        protein, reference_folder, target_folder, metadata_file, gradient_ref_index
    )

    # the value that drives the shared colour gradient: TM-score to the chosen
    # reference. Panel x/y for the TM plot are the two references' scores.
    gradient_col = "tm_state_1" if gradient_ref_index == 0 else "tm_state_2"
    gradient_label = reference_labels[gradient_ref_index]

    if gradient_range is not None:
        vmin, vmax = gradient_range
    else:
        vmin, vmax = float(df[gradient_col].min()), float(df[gradient_col].max())
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # Per-point fill: the TM-score gradient when shading is on, else one neutral
    # colour for every point (shape still tells the models apart).
    _NO_SHADE_COLOR = "lightgray"

    def point_colors(sub: pd.DataFrame):
        return cmap(norm(sub[gradient_col])) if color_by_tm else _NO_SHADE_COLOR

    models = sorted(df["model"].unique())

    # The two panels deliberately have different shapes: the TM-score plot is
    # square (its two axes are the same kind of quantity), while the PCA plot is
    # wider than tall (PC1 carries far more variance than PC2). Give PCA more
    # horizontal room via width_ratios, then pin each panel's box shape with
    # set_box_aspect below so the ratio holds regardless of label sizes.
    fig, (ax_tm, ax_pca) = plt.subplots(
        1, 2, figsize=(15, 6.5), dpi=300,
        gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.25},
    )

    # ---- left: TM-score scatter, coloured by gradient, shaped by model --------
    ax_tm.grid(True, color="lightgray", linewidth=0.5, alpha=0.4)
    for model in models:
        sub = df[df["model"] == model]
        ax_tm.scatter(
            sub["tm_state_1"],
            sub["tm_state_2"],
            c=point_colors(sub),
            marker=model_marker(model),
            s=70,
            edgecolors="black",
            linewidths=0.5,
        )

    if plot_guidelines:
        reference_tm, _, _ = __calc_tm_score(
            *get_reference_files(
                protein, reference_folder.rstrip("/") + "/", metadata_file
            )
        )
        ax_tm.axvline(x=reference_tm, c="gray", linestyle="--", linewidth=1)
        ax_tm.axhline(y=reference_tm, c="gray", linestyle="--", linewidth=1)

    ax_tm.set_xlabel(
        f"Similarity to {reference_labels[0]} conformation (TM-score)",
        fontsize=font_size,
        fontweight="bold",
    )
    ax_tm.set_ylabel(
        f"Similarity to {reference_labels[1]} conformation (TM-score)",
        fontsize=font_size,
        fontweight="bold",
    )
    ax_tm.set_title(
        f"{experiment_name} | {protein}",
        fontsize=font_size + 2,
        fontweight="bold",
    )
    ax_tm.tick_params(axis="both", labelsize=font_size)
    if limit_axis:
        try:
            low = get_axis_lower_limit(protein)
            ax_tm.set_xlim(low, 1)
            ax_tm.set_ylim(low, 1)
        except KeyError:
            warnings.warn(f"No axis limit known for {protein}; using auto limits")
    # square panel: both axes are TM-scores on the same scale
    ax_tm.set_box_aspect(1)

    # ---- right: PCA scatter, same gradient + same shapes ---------------------
    ax_pca.grid(True, color="lightgray", linewidth=0.5, alpha=0.4)
    for model in models:
        sub = df[df["model"] == model]
        ax_pca.scatter(
            sub["pc1"],
            sub["pc2"],
            c=point_colors(sub),
            marker=model_marker(model),
            s=70,
            edgecolors="black",
            linewidths=0.5,
        )
    ax_pca.set_xlabel(
        f"PC1 ({100 * explained[0]:.1f}%)", fontsize=font_size, fontweight="bold"
    )
    ax_pca.set_ylabel(
        f"PC2 ({100 * explained[1]:.1f}%)", fontsize=font_size, fontweight="bold"
    )
    ax_pca.set_title(
        f"{experiment_name} | {protein} | PCA of Cα coordinates",
        fontsize=font_size + 2,
        fontweight="bold",
    )
    ax_pca.tick_params(axis="both", labelsize=font_size)
    # rectangular panel, wider than tall (PC1 >> PC2 variance)
    ax_pca.set_box_aspect(0.6)

    # ---- shared colorbar (the gradient, describing both panels) --------------
    # Only when the TM-score shading is on. Give the colorbar its own axes pinned
    # just to the right of the PCA panel (in PCA axes-fraction coords, so it
    # tracks the panel's box and never sits on top of the data), matching the
    # panel's height.
    if color_by_tm:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = ax_pca.inset_axes([1.04, 0.0, 0.035, 1.0])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(
            f"TM-score to {gradient_label} conformation",
            fontweight="bold",
            fontsize=font_size,
        )
        cbar.ax.tick_params(labelsize=font_size)

    # ---- combined shape legend (shape -> model), shared by both panels -------
    model_handles = [
        Line2D(
            [0],
            [0],
            marker=model_marker(m),
            linestyle="None",
            markerfacecolor="lightgray",
            markeredgecolor="black",
            markersize=font_size * 0.6,
            label=f"Model {m}",
        )
        for m in models
    ]
    # Bottom-align both panels in their cells: set_box_aspect leaves the short
    # PCA panel centred with whitespace beneath it, which would push the legend
    # far down. Anchoring "S" drops both panels' bottoms to the same line so the
    # legend can sit just under them.
    ax_tm.set_anchor("S")
    ax_pca.set_anchor("S")

    # loc="upper center" anchors by the legend's TOP edge, so bbox y is simply
    # where the top of the legend sits (figure fraction) -- just below the panels'
    # x-axis labels. Raise y to tighten the gap, lower it to loosen.
    fig.legend(
        handles=model_handles,
        title="AlphaFold model",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=len(model_handles),
        prop={"size": font_size, "weight": "bold"},
        title_fontproperties={"weight": "bold", "size": font_size + 1},
    )

    fig.subplots_adjust(bottom=0.18)

    output_dir = output_dir.rstrip("/") + "/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if save_file_name is None:
        save_file_name = f"{protein}_{experiment_name}_tm_pca.png"
    output_path = output_dir + save_file_name

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  [green]done[/] -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protein", type=str, required=True)
    parser.add_argument("--reference_folder", type=str, required=True)
    parser.add_argument("--target_folder", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument(
        "--gradient_reference",
        type=str,
        choices=["state_1", "state_2"],
        default="state_1",
        help="Which conformation the colour gradient is measured against",
    )
    parser.add_argument(
        "--gradient_range",
        nargs=2,
        type=float,
        metavar=("VMIN", "VMAX"),
        default=None,
        help="Fix the gradient (colorbar) axis, e.g. --gradient_range 0.5 1.0, "
        "so several plots share one colour scale; default: this plot's min/max",
    )
    parser.add_argument("--save_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--font_size", type=int, default=12)
    parser.add_argument(
        "--limit_axis",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Limit the TM-score axes to the protein's known lower bound..1",
    )
    parser.add_argument(
        "--guidelines",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Draw the reference-vs-reference TM-score guide lines on the TM panel",
    )
    parser.add_argument(
        "--color_by_tm",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Shade points by TM-score to the gradient reference (with colorbar); "
        "false turns shading off entirely -- one neutral fill, no colorbar",
    )
    args = parser.parse_args()

    plot_tm_pca(
        protein=args.protein,
        reference_folder=args.reference_folder,
        target_folder=args.target_folder,
        metadata_file=args.metadata_path,
        experiment_name=args.experiment_name,
        gradient_reference=args.gradient_reference,
        gradient_range=tuple(args.gradient_range) if args.gradient_range else None,
        save_file_name=args.save_file,
        output_dir=args.output_dir,
        font_size=args.font_size,
        limit_axis=args.limit_axis,
        plot_guidelines=args.guidelines,
        color_by_tm=args.color_by_tm,
    )


if __name__ == "__main__":
    main()
