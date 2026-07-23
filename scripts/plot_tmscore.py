import argparse
import json
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Union, List

import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

console = Console()

# Standard annotation font sizes for TM-score plots, kept here as the single
# source of truth so everything that reuses this module -- plot_tm_pca,
# cluster_conformations, ... -- renders axis titles and legends at the same size.
# Spread into a plot_tm_score(...) call via ``**TM_ANNOTATION_FONTSIZES``, or read
# an individual value when drawing a TM-score panel by hand.
TM_ANNOTATION_FONTSIZES = {
    "axis_title_fontsize": 14,
    "legend_text_fontsize": 12,
    "legend_title_fontsize": 12,
}


def _natural_sort_key(value):
    """
    Sort key that splits a value into text/number chunks so strings like
    'depth_16', 'depth_32', 'depth_128' sort in numeric order
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [value]
    s = str(value)
    return [
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in re.split(r"(\d+)", s)
    ]


def __calc_tm_score(file1_path: str, file2_path: str) -> tuple[float, int, int]:
    """
    :param file1_path: path to .pdb file
    :param file2_path: path to .pdb file

    Calculate TM-score between proteins in file1 and file2.
    Return list of TM-scores between proteins in file1 and file2 and their sequence lengths.
    """
    s1 = get_structure(file1_path)
    s2 = get_structure(file2_path)

    coords1, seq1 = get_residue_data(next(s1.get_chains()))
    coords2, seq2 = get_residue_data(next(s2.get_chains()))

    len1, len2 = len(seq1), len(seq2)

    tm_score = tm_align(coords1, coords2, seq1, seq2).tm_norm_chain1
    return tm_score, len1, len2


def get_reference_files(protein: str, reference_folder: str, metadata_file: str) -> list:
    """
    :param protein: protein name
    :param reference_folder: path to folder containing reference files
    :param metadata_file: path to metadata file

    Return list of reference files
    """
    metadata = json.load(open(metadata_file))

    IF_ID = metadata[protein]["conformations"]["state_1"]["pdb_id"]
    IF_CHAIN = metadata[protein]["conformations"]["state_1"]["chain"]
    IF_LABEL = metadata[protein]["conformations"]["state_1"]["label"]
    OF_ID = metadata[protein]["conformations"]["state_2"]["pdb_id"]
    OF_CHAIN = metadata[protein]["conformations"]["state_2"]["chain"]
    OF_LABEL = metadata[protein]["conformations"]["state_2"]["label"]

    return [
        reference_folder + f"{IF_LABEL}_{IF_ID}_{IF_CHAIN}.pdb",
        reference_folder + f"{OF_LABEL}_{OF_ID}_{OF_CHAIN}.pdb"
    ]


def calc_tm_score_folders(
        protein: str,
        reference_folder: str,
        target_folder: str,
        metadata_file: str,
        output_file_name: str,
        output_dir: str,
        model:str = None,
        seed:str = None
) -> str:
    """
    :param protein: protein name
    :param reference_folder: path to folder containing reference files
    :param target_folder: path to folder containing target files
    :param metadata_file: path to metadata file
    :param output_file_name: name of output csv file
    :param output_dir: path to output folder
    :param model: AF model
    :param seed: AF model seed

    Calculate TM-score from folder of reference files and folder of target files,
    and return the result as csv file.

    Return csv file schema:
    output_file_name(protein, nseq, tm_A, tm_I) if functional
    or
    output_file_name(protein, nseq, tm_OF, tm_IF) if conformational
    """
    reference_folder = reference_folder + "/" if reference_folder[-1] != "/" else reference_folder
    target_folder = target_folder + "/" if target_folder[-1] != "/" else target_folder

    nseq = json.load(open(target_folder + "config.json"))["max_extra_seq"]

    reference_files = get_reference_files(protein, reference_folder, metadata_file)
    target_files = [str(f) for f in Path(target_folder).glob(
        f"{protein}_unrelaxed_*_alphafold2_model_{'*' if model is None else model}_seed_{'*' if seed is None else seed}.pdb"
    )]

    if len(target_files) != 25:
        warnings.warn(f"There are {len(target_files)} in {target_folder}, instead of the expected 25!")

    results_df = pd.DataFrame()
    reference_tm, _, _ = __calc_tm_score(reference_files[0], reference_files[1])
    progress = Progress(
        SpinnerColumn(),
        TextColumn(f"  {protein} TM-score"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("models"),
        console=console,
    )

    with progress:
        task = progress.add_task(protein, total=len(target_files))
        for target_file in target_files:
            # Skip files whose model/seed can't be parsed as integers, e.g.
            # Google-Drive re-download duplicates like "..._seed_000(1).pdb".
            try:
                seed_value = int(target_file.split("_")[-1].split(".")[0])
                model_value = int(target_file.split("_")[-3])
            except ValueError:
                warnings.warn(
                    f"Skipping {target_file}: unexpected file name, cannot parse model/seed."
                )
                continue

            row = {
                "protein": protein,
                "nseq": nseq,
                "reference_tm": reference_tm,
                "seed": seed_value,
                "model": model_value,
                # basename of the experiment output folder; use os.path so this
                # is correct on Windows (backslash paths) as well as POSIX.
                "experiment": os.path.basename(os.path.normpath(output_dir))
            }
            for reference_file in reference_files:
                tm_score, len_seq1, len_seq2 = __calc_tm_score(
                    reference_file,
                    target_file
                )
                row["len_seq1"] = len_seq1
                row["len_seq2"] = len_seq2

                ref_type = reference_file.split("/")[-1].split("_")[0].lower()
                if "active" == ref_type:
                    row["tm_A"] = tm_score
                elif "inactive" == ref_type:
                    row["tm_I"] = tm_score
                elif "of" == ref_type:
                    row["tm_OF"] = tm_score
                elif "if" == ref_type:
                    row["tm_IF"] = tm_score
                else:
                    raise ValueError(f"File {reference_file} not recognized")
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        progress.advance(task)

    if output_dir is not None:
        if output_dir[-1] != "/":
            output_dir += "/"
        if output_file_name is not None:
            output_file_name = output_dir + output_file_name
    output_file_name = output_file_name if output_file_name is not None else f"{output_dir}{protein}.csv"
    results_df.to_csv(output_file_name, index=False)
    return output_file_name


def get_axis_lower_limit(protein: str) -> float:
    return {
        "CGRPR": 0.82,
        "FZD7": 0.85,
        "PTH1R": 0.75,
        "ASCT2": 0.55,
        "STP10": 0.8,
        "LAT1": 0.85,
        "ZnT8": 0.73,
        "MCT1": 0.70,
        "CCR5": 0.88,
        "MurJ": 0.65,
        "PfMATE": 0.65,
        "SERT": 0.85
    }[protein]


def depth_color(depth: int, colormap: str = "okabe_ito"):
    values = [16, 32, 64, 128, 256, 512, 1024, 5120]
    cmap = plt.get_cmap(colormap)
    positions = np.linspace(0, 1, len(values))
    color_dict = {
        v: mcolors.to_hex(cmap(p))
        for v, p in zip(sorted(values), positions)
    }
    return color_dict[depth]


def get_okabe_ito_colors() -> list:
    """
    Return the Okabe-Ito colour-blind-safe palette as a list of hex strings.

    Handy for the ``colors=`` argument of ``plot_tm_score`` / ``combine_plots``:
    pass palette entries directly or mix them with your own colours, e.g.
    ``colors=[get_okabe_ito_colors()[0], "#000000"]``.
    """
    cmap = plt.get_cmap("okabe_ito")
    return [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]


def _arrange_legend(handles: list, legend_layout: str):
    """
    Return ``(handles, ncol)`` to pass to ``ax.legend`` / ``figure.legend``.

    - ``"row"``    -> one horizontal row.
    - ``"column"`` -> one vertical column.
    - ``"auto"``   -> a compact, ~square grid that **reads left-to-right** (row-major),
      e.g. 4 -> 2x2, 3 -> 2 + 1, 5 -> 3 + 2.

    Matplotlib fills legends column-major, so for ``"auto"`` we pad to a full grid with
    invisible entries and reshuffle into column-major order. The blank cells then land at
    the end of the last row (where a row-major grid expects them), which keeps the visible
    entries in label order regardless of how many there are.
    """
    n = len(handles)
    if legend_layout == "column":
        return handles, 1
    if legend_layout == "row" or n <= 1:
        return handles, max(n, 1)

    # "auto": compact grid, row-major reading order
    ncol = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncol))
    blank = Line2D([], [], linestyle="", marker="", label="")
    padded = list(handles) + [blank] * (nrows * ncol - n)
    ordered = [padded[(k % nrows) * ncol + (k // nrows)] for k in range(nrows * ncol)]
    return ordered, ncol


def _round_ticks_with_one(ax, axis: str, nbins: int = 6):
    """
    Set nicely-rounded ticks on the given axis ('x' or 'y') and make sure
    1.0 is always included as a labeled tick, as long as it's within the
    current axis limits.
    """
    if axis == "x":
        get_lim, set_ticks = ax.get_xlim, ax.set_xticks
    else:
        get_lim, set_ticks = ax.get_ylim, ax.set_yticks

    locator = MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10])
    lo, hi = get_lim()
    ticks = locator.tick_values(lo, hi)
    ticks = ticks[(ticks >= lo) & (ticks <= hi)]

    if lo <= 1 <= hi and not np.any(np.isclose(ticks, 1.0)):
        ticks = np.append(ticks, 1.0)

    ticks = np.unique(np.round(ticks, 10))
    set_ticks(ticks)


def plot_tm_score(
        data_file: str,
        save_file_name: str = None,
        protein: str = None,
        title: str = None,
        limit_axis: bool = True,
        output_dir: str = None,
        experiment_name: str = None,
        axis_bounds: tuple[float, float, float, float] = None,
        plot_guidelines = True,
        font_size: int = 6,
        opacity: float = 1,
        color_on: str = None,
        shape_on: str = None,
        legend_font_size: int = None,
        literal_colors: bool = False,
        legend_entries: list = None,
        legend_title: str = None,
        colormap: str = "okabe_ito",
        legend_labels: list = None,
        legend_layout: str = "auto",
        tick_anchor: float = None,
        tick_size: float = None,
        x_axis_title: str = None,
        y_axis_title: str = None,
        axis_title_fontsize: float = None,
        legend_title_fontsize: float = None,
        legend_text_fontsize: float = None,
        colors: list = None,
        show_title: bool = True,
        legend_mode: str = "inline"
) -> None:
    """
    :param data_file: path to csv file
    :param save_file_name: name of output image file
    :param protein: protein name if more than 1 protein in csv
    :param title: title of the plot
    :param limit_axis: if the axis should be lower limited
    :param output_dir: directory to save the plot
    :param experiment_name: name of the experiment
    :param axis_bounds: [l_x, h_x, l_y, h_y]
            where:
            - l_x: lower x bound
            - h_x: upper x bound
            - l_y: lower y bound
            - h_y: upper y bound
    :param plot_guidelines: whether to plot guidelines with IF/OF TM score
    :param font_size: font size
    :param opacity of the points on the scatterplot
    :param color_on: column to color points by
    :param shape_on: column to shape points by
    :param legend_font_size: font size for the categorical color legend only
            (markers scale with it); defaults to font_size + 1
    :param literal_colors: if True, the ``color_on`` column already holds
            matplotlib colour strings that are used verbatim per point (no
            colormap / colorbar). Useful for per-point shading.
    :param legend_entries: optional list of ``(label, color)`` pairs; when given
            (typically with ``literal_colors``) a compact legend of these is
            drawn instead of the automatic colorbar/category legend.
    :param legend_title: title for the ``legend_entries`` legend.
    :param colormap: name of the matplotlib colormap to use for coloring
            points (e.g. "okabe_ito", "viridis", "plasma", ...)
    :param legend_labels: optional list of labels for the categorical-colour legend,
            aligned to the sorted colour categories; overrides the raw category values.
    :param legend_layout: arrangement of the discrete legend entries:
            "auto" (compact ~square grid, row-major), "row", or "column".
    :param tick_anchor, tick_size: when both are given, place ticks on x and y at
            ``tick_anchor + k * tick_size`` across the (limited) axis range, overriding
            the default rounded-ticks locator.
    :param x_axis_title, y_axis_title: override the default axis-label text.
    :param axis_title_fontsize: absolute font size for the x/y axis titles
            (defaults to font_size + 1).
    :param legend_text_fontsize: absolute font size for legend entry text
            (defaults to legend_font_size, else font_size + 1).
    :param legend_title_fontsize: absolute font size for the legend title
            (defaults to legend_text_fontsize + 1).
    :param colors: optional palette (list of colour strings) for categorical colouring;
            replaces the colormap-derived colours. See ``get_okabe_ito_colors``.
    :param show_title: whether to draw the plot title.
    :param legend_mode: how to render the discrete (shape/categorical) legend(s):
            "inline" draws them on the plot, "separate" exports each to a
            "<name>_legend<ext>" file instead, "none" omits them.

    Scatterplot of IF-OF / inactive-active TM-scores for different MSA depths.
    """
    try:
        data = pd.read_csv(data_file)
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"No columns to parse from file: {data_file}") from e

    if protein is None:
        assert data["protein"].unique().size == 1
    else:
        data = data[data["protein"] == protein]

    protein = data["protein"].unique()[0]

    color_on = "nseq" if color_on is None else color_on

    assert ("tm_IF" in data.columns and "tm_OF" in data.columns) or ("tm_A" in data.columns and "tm_I" in data.columns)
    structure_type = "conformational" if "tm_IF" in data.columns else "functional"

    if structure_type == "conformational":
        x_col_name = "tm_IF"
        y_col_name = "tm_OF"
    else:
        x_col_name = "tm_I"
        y_col_name = "tm_A"

    data = data.sort_values(color_on, key=lambda col: col.map(_natural_sort_key))

    assert legend_layout in {"auto", "row", "column"}, \
        f"legend_layout must be 'auto', 'row', or 'column', got {legend_layout!r}"
    assert legend_mode in {"inline", "separate", "none"}, \
        f"legend_mode must be 'inline', 'separate', or 'none', got {legend_mode!r}"

    # Resolve absolute font sizes; fall back to the historical font_size + N behaviour.
    # legend_text_fontsize supersedes the older legend_font_size knob when both are given.
    legend_text_size = (
        legend_text_fontsize if legend_text_fontsize is not None
        else legend_font_size if legend_font_size is not None
        else font_size + 1
    )
    legend_title_size = (
        legend_title_fontsize if legend_title_fontsize is not None else legend_text_size + 1
    )
    axis_title_size = axis_title_fontsize if axis_title_fontsize is not None else font_size + 1
    legend_marker_size = max(legend_text_size - 3, 2)

    cmap_obj = plt.get_cmap(colormap)

    # Non-numeric color columns (e.g. "experiment") can't be shown on a colorbar,
    # so they get a discrete color per category and a legend instead.
    # When literal_colors is set, the column holds ready-made colours; it is
    # neither categorical nor numeric-mapped.
    color_is_categorical = (
        not literal_colors
        and color_on != "nseq"
        and not pd.api.types.is_numeric_dtype(data[color_on])
    )
    if color_is_categorical:
        color_categories = sorted(data[color_on].unique(), key=_natural_sort_key)
        # An explicit `colors` palette overrides the colormap-derived colours.
        palette = colors if colors is not None else [
            mcolors.to_hex(cmap_obj(i % cmap_obj.N)) for i in range(len(color_categories))
        ]
        category_color_map = {
            cat: palette[i % len(palette)]
            for i, cat in enumerate(color_categories)
        }

    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.grid(True, color="lightgray", linewidth=0.5, alpha=0.4)

    # Discrete legends (shape and/or categorical colour) are collected here and rendered
    # after the plot according to `legend_mode`. Colourbars are always drawn inline.
    legend_specs = []

    if shape_on is not None:
        markers = [
            "o", "s", "^", "D", "v", "<", ">", "P", "X",
            "*", "h", "H", "8", "p", "d"
        ]

        categories = sorted(data[shape_on].dropna().unique(), key=_natural_sort_key)
        marker_map = {cat: markers[i % len(markers)] for i, cat in enumerate(categories)}

        for cat in categories:
            mask = data[shape_on] == cat
            if color_is_categorical:
                point_colors = [category_color_map[v] for v in data.loc[mask, color_on]]
            elif color_on == "nseq":
                point_colors = [depth_color(d, colormap) for d in data.loc[mask, color_on]]
            else:
                point_colors = data.loc[mask, color_on]

            scatter = ax.scatter(
                data.loc[mask, x_col_name],
                data.loc[mask, y_col_name],
                c=point_colors,
                marker=marker_map[cat],
                s=30,
                edgecolors="black",
                linewidths=0.375,
                alpha=opacity
            )

        shape_handles = [
            Line2D(
                [0], [0],
                marker=marker,
                linestyle="",
                markerfacecolor="lightgray",
                markeredgecolor="black",
                markersize=6,
                label=str(cat),
            )
            for cat, marker in marker_map.items()
        ]

        shape_handles, shape_ncol = _arrange_legend(shape_handles, legend_layout)
        legend_specs.append({
            "handles": shape_handles,
            "title": f"{shape_on} shape values:",
            "ncol": shape_ncol,
            "loc": "upper center",
            "bbox_to_anchor": (0.5, -0.15),
        })
    elif color_is_categorical:
        # One scatter per category so `opacity` may be a scalar or a list/tuple
        # aligned to the SORTED categories (same convention as `colors`).
        for i, cat in enumerate(color_categories):
            mask = data[color_on] == cat
            cat_alpha = opacity[i] if isinstance(opacity, (list, tuple, np.ndarray)) else opacity
            ax.scatter(
                data.loc[mask, x_col_name],
                data.loc[mask, y_col_name],
                c=category_color_map[cat],
                s=30,
                edgecolors="black",
                linewidths=0.375,
                alpha=cat_alpha,
            )
    else:
        if literal_colors:
            c_vals = list(data[color_on])
            cmap, norm = None, None
        elif color_on == "nseq":
            c_vals = [depth_color(d, colormap) for d in data[color_on]]
            cmap, norm = None, None
        else:
            unique_vals = sorted(data[color_on].unique(), key=_natural_sort_key)
            c_vals = [unique_vals.index(v) for v in data[color_on]]
            cmap = colormap
            norm = mcolors.BoundaryNorm(np.arange(-0.5, len(unique_vals) + 0.5), cmap_obj.N)

        scatter = ax.scatter(
            data[x_col_name],
            data[y_col_name],
            c=c_vals,
            s=30,
            edgecolors="black",
            linewidths=0.375,
            alpha=opacity,
            cmap=cmap,
            norm=norm
        )

    if structure_type == "conformational":
        x_label = "IF Conf."
        y_label = "OF Conf."
    else:
        x_label = "Inactive Conf."
        y_label = "Active Conf."

    xlabel = x_axis_title if x_axis_title is not None else f"TM-Score: Pred vs {x_label}"
    ylabel = y_axis_title if y_axis_title is not None else f"TM-Score: Pred vs {y_label}"

    ax.set_xlabel(
        xlabel,
        fontsize=axis_title_size,
        labelpad=font_size + 4,
        fontweight="bold"
    )

    ax.set_ylabel(
        ylabel,
        fontsize=axis_title_size,
        labelpad=font_size + 4,
        fontweight="bold"
    )

    ax.tick_params(axis="both", labelsize=font_size + 3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("black")

    if plot_guidelines:
        reference_tm = float(data.loc[0]["reference_tm"])
        plt.axvline(x=reference_tm, c="gray", linestyle="--")
        plt.axhline(y=reference_tm, c="gray", linestyle="--")

    unique_depths = sorted(data[color_on].unique(), key=_natural_sort_key)

    if literal_colors:
        depth_text = f"{', '.join([str(x) for x in np.sort(data['nseq'].unique()).tolist()])}"
        if legend_entries:
            handles = [
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markersize=legend_marker_size,
                    label=str(label),
                )
                for label, color in legend_entries
            ]
            handles, entries_ncol = _arrange_legend(handles, legend_layout)
            legend_specs.append({
                "handles": handles,
                "title": legend_title if legend_title is not None else "",
                "ncol": entries_ncol,
                "loc": "center left",
                "bbox_to_anchor": (1.02, 0.5),
            })
    elif color_on == "nseq":
        if len(unique_depths) > 1:
            depth_text = ", ".join([str(x) for x in unique_depths])

            used_colors = [depth_color(d, colormap) for d in unique_depths]
            legend_cmap = mcolors.ListedColormap(used_colors)

            bounds = np.arange(len(unique_depths) + 1)
            legend_norm = mcolors.BoundaryNorm(bounds, legend_cmap.N)

            sm = plt.cm.ScalarMappable(cmap=legend_cmap, norm=legend_norm)
            sm.set_array([])

            cbar = plt.colorbar(sm, ax=ax)

            cbar.set_label(
                "MSA depth (sequences)",
                fontsize=font_size + 1,
                fontweight="bold",
                labelpad=font_size + 1
            )

            cbar.ax.tick_params(labelsize=font_size + 2)
            for tick in cbar.ax.get_yticklabels():
                tick.set_fontweight('bold')

            cbar.set_ticks(np.arange(len(unique_depths)) + 0.5)
            cbar.ax.tick_params(which="major", length=0)
            cbar.set_ticklabels([str(int(d)) for d in unique_depths])

            cbar.ax.minorticks_off()
            cbar.ax.tick_params(which="minor", length=0)
        else:
            depth_text = f"{unique_depths[0]}"
    elif color_is_categorical:
        sorted_nseq = sorted(data["nseq"].unique(), key=_natural_sort_key)
        depth_text = f"{', '.join([str(x) for x in sorted_nseq])}"

        color_handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor=category_color_map[cat],
                markeredgecolor="black",
                markersize=legend_marker_size,
                label=(
                    legend_labels[i]
                    if legend_labels is not None and i < len(legend_labels)
                    else str(cat)
                ),
            )
            for i, cat in enumerate(color_categories)
        ]

        # A shape legend may already have been collected; both are rendered together
        # below (see `legend_specs`), so a shape + colour legend stay visible at once.
        color_handles, color_ncol = _arrange_legend(color_handles, legend_layout)
        legend_specs.append({
            "handles": color_handles,
            "title": legend_title if legend_title is not None else f"{color_on}:",
            "ncol": color_ncol,
            "loc": "center left",
            "bbox_to_anchor": (1.02, 0.5),
        })
    else:
        sorted_nseq = sorted(data["nseq"].unique(), key=_natural_sort_key)
        depth_text = f"{', '.join([str(x) for x in sorted_nseq])}"
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(
            color_on,
            fontsize=font_size + 1,
            fontweight="bold",
            labelpad=font_size + 1
        )
        unique_vals = sorted(data[color_on].unique(), key=_natural_sort_key)
        cbar.set_ticks(np.arange(len(unique_vals)))
        cbar.set_ticklabels([str(int(v)) for v in unique_vals])

    if limit_axis:
        if axis_bounds is None:
            ax.set_xlim(get_axis_lower_limit(protein), 1)
            ax.set_ylim(get_axis_lower_limit(protein), 1)
        else:
            ax.set_xlim(axis_bounds[0], axis_bounds[1])
            ax.set_ylim(axis_bounds[2], axis_bounds[3])
    ax.set_aspect("auto", adjustable="box")

    # Custom tick grid aligned to `tick_anchor` with spacing `tick_size`, filling the
    # whole (already-limited) axis range on both x and y; otherwise use the default
    # rounded-ticks locator that always keeps 1.0 labelled.
    if tick_anchor is not None and tick_size is not None:
        for set_ticks, lim in [(ax.set_xticks, ax.get_xlim()), (ax.set_yticks, ax.get_ylim())]:
            low, high = min(lim), max(lim)
            k_min = int(np.ceil((low - tick_anchor) / tick_size))
            k_max = int(np.floor((high - tick_anchor) / tick_size))
            ticks = np.round(tick_anchor + tick_size * np.arange(k_min, k_max + 1), 8)
            set_ticks(ticks)
    else:
        _round_ticks_with_one(ax, "x")
        _round_ticks_with_one(ax, "y")

    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

    if show_title:
        if title is not None:
            plt.title(title, fontsize=font_size + 5, weight="bold", pad=font_size + 7)
        else:
            title = f"{protein} | Depth: {depth_text}"
            if experiment_name is not None:
                title = "Experiment: " + experiment_name + "\n" + title
            plt.title(title, fontsize=font_size + 5, weight="bold", pad=font_size + 7)

    if output_dir is not None:
        if output_dir[-1] != "/":
            output_dir += "/"
        if save_file_name is not None:
            save_file_name = output_dir + save_file_name
    save_file_name = save_file_name if save_file_name is not None else f"{output_dir}{protein}.png"

    def _legend_kwargs(spec):
        return dict(
            handles=spec["handles"],
            title=spec["title"],
            ncol=spec["ncol"],
            fontsize=legend_text_size,
            title_fontproperties=FontProperties(weight="bold", size=legend_title_size),
        )

    # "inline": draw the legend(s) on the plot. Multiple legends on one axes require
    # add_artist on all but the last, otherwise each ax.legend() overwrites the previous.
    if legend_specs and legend_mode == "inline":
        for i, spec in enumerate(legend_specs):
            leg = ax.legend(
                loc=spec["loc"],
                bbox_to_anchor=spec["bbox_to_anchor"],
                **_legend_kwargs(spec),
            )
            if i != len(legend_specs) - 1:
                ax.add_artist(leg)

    fig.savefig(
        save_file_name,
        dpi=300,
        bbox_inches="tight"
    )

    # "separate": export each legend to its own "<name>_legend[_i]<ext>" file.
    if legend_specs and legend_mode == "separate":
        p = Path(save_file_name)
        for i, spec in enumerate(legend_specs):
            suffix = "_legend" if len(legend_specs) == 1 else f"_legend_{i}"
            legend_path = str(p.with_name(f"{p.stem}{suffix}{p.suffix}"))
            fig_legend = plt.figure(figsize=(4, 3), dpi=300)
            fig_legend.legend(loc="center", **_legend_kwargs(spec))
            fig_legend.savefig(legend_path, dpi=300, bbox_inches="tight")
            plt.close(fig_legend)
            console.print(f"  [green]saved legend[/] {legend_path}")

    plt.close(fig)


def combine_plots(
        data_files: Union[List [str], str],
        save_file_name: str = None,
        protein: str = None,
        title: str = None,
        limit_axis: bool = True,
        output_dir: str = None,
        experiment_name: str = None,
        axis_bounds: tuple[float, float, float, float] = None,
        plot_guidelines = True,
        font_size: int = 6,
        opacity: float = 1,
        color_on: str = None,
        shape_on: str = None,
        colormap: str = "okabe_ito",
        legend_font_size: int = None,
        literal_colors: bool = False,
        legend_entries: list = None,
        legend_title: str = None,
        legend_labels: list = None,
        legend_layout: str = "auto",
        tick_anchor: float = None,
        tick_size: float = None,
        x_axis_title: str = None,
        y_axis_title: str = None,
        axis_title_fontsize: float = None,
        legend_title_fontsize: float = None,
        legend_text_fontsize: float = None,
        colors: list = None,
        show_title: bool = True,
        legend_mode: str = "inline"
) -> None:
    """
    :param colormap: name of the matplotlib colormap to use for coloring
            points (e.g. "okabe_ito", "viridis", "plasma", ...), forwarded
            to plot_tm_score for the combined plot.

    All other keyword arguments are forwarded verbatim to ``plot_tm_score``; see
    its docstring for the styling knobs (font sizes, legend layout/mode, etc.).
    """
    dfs = []
    ref_cols = None

    for f in data_files:
        df = pd.read_csv(f)

        if ref_cols is None:
            ref_cols = df.columns
        else:
            assert set(df.columns) == set(ref_cols), \
                f"Columns of {f} do not match"

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    if output_dir is not None:
        out = output_dir if output_dir.endswith("/") else output_dir + "/"
        os.makedirs(out, exist_ok=True)
        base = protein if protein is not None else "combined"
        filename = f"{out}{base}_combined.csv"
    else:
        filename = (
            f"{('').join(data_files[0].split(".")[:-1])}_COMBINED_"
            f"{datetime.now().strftime('%m_%d_%H_%M')}.csv"
        )

    combined_df.to_csv(filename, index=False)
    print("Saved COMBINED file to", filename)

    plot_tm_score(
        data_file=filename,
        save_file_name=save_file_name,
        protein=protein,
        title=title,
        limit_axis=limit_axis,
        output_dir=output_dir,
        experiment_name=experiment_name,
        axis_bounds=axis_bounds,
        plot_guidelines=plot_guidelines,
        font_size=font_size,
        opacity=opacity,
        color_on=color_on,
        shape_on=shape_on,
        colormap=colormap,
        legend_font_size=legend_font_size,
        literal_colors=literal_colors,
        legend_entries=legend_entries,
        legend_title=legend_title,
        legend_labels=legend_labels,
        legend_layout=legend_layout,
        tick_anchor=tick_anchor,
        tick_size=tick_size,
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        axis_title_fontsize=axis_title_fontsize,
        legend_title_fontsize=legend_title_fontsize,
        legend_text_fontsize=legend_text_fontsize,
        colors=colors,
        show_title=show_title,
        legend_mode=legend_mode,
    )


def _add_style_args(parser):
    """Shared presentation/styling flags for the plotting subcommands."""
    parser.add_argument("--axis_title_fontsize", type=float, default=None)
    parser.add_argument("--legend_title_fontsize", type=float, default=None)
    parser.add_argument("--legend_text_fontsize", type=float, default=None)
    parser.add_argument("--legend_title", type=str, default=None)
    parser.add_argument("--legend_labels", nargs="+", type=str, default=None)
    parser.add_argument(
        "--legend_layout", type=str, default="auto", choices=["auto", "row", "column"]
    )
    parser.add_argument(
        "--legend_mode", type=str, default="inline", choices=["inline", "separate", "none"]
    )
    parser.add_argument("--x_axis_title", type=str, default=None)
    parser.add_argument("--y_axis_title", type=str, default=None)
    parser.add_argument("--tick_anchor", type=float, default=None)
    parser.add_argument("--tick_size", type=float, default=None)
    parser.add_argument("--colors", nargs="+", type=str, default=None)
    parser.add_argument(
        "--show_title", type=lambda x: x.lower() == "true", default=True
    )


def _style_kwargs(args):
    """Collect the shared styling flags into kwargs for plot_tm_score/combine_plots."""
    return dict(
        legend_title=args.legend_title,
        legend_labels=args.legend_labels,
        legend_layout=args.legend_layout,
        tick_anchor=args.tick_anchor,
        tick_size=args.tick_size,
        x_axis_title=args.x_axis_title,
        y_axis_title=args.y_axis_title,
        axis_title_fontsize=args.axis_title_fontsize,
        legend_title_fontsize=args.legend_title_fontsize,
        legend_text_fontsize=args.legend_text_fontsize,
        colors=args.colors,
        show_title=args.show_title,
        legend_mode=args.legend_mode,
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    calc_parser = subparsers.add_parser("calc", help="Calculate TM-scores and save CSV")
    calc_parser.add_argument("--protein", type=str, required=True)
    calc_parser.add_argument("--reference_folder", type=str, required=True)
    calc_parser.add_argument("--target_folder", type=str, required=True)
    calc_parser.add_argument("--output_file", type=str, default=None)
    calc_parser.add_argument("--metadata_path", type=str, required=True)
    calc_parser.add_argument("--output_dir", type=str, default=None)

    combine_parser = subparsers.add_parser("combine", help="Combine files and plot TM-scores")
    combine_parser.add_argument(
        "--data_files",
        nargs="+",
        type=str,
        required=True,
        help="One or more CSV files"
    )
    combine_parser.add_argument("--save_file", type=str, default=None)
    combine_parser.add_argument("--protein", type=str, default=None)
    combine_parser.add_argument("--title", type=str, default=None)
    combine_parser.add_argument("--model", type=str, default=None)
    combine_parser.add_argument("--seed", type=str, default=None)
    combine_parser.add_argument(
        "--limit_axis",
        type=lambda x: x.lower() == "true",
        default=True
    )
    combine_parser.add_argument("--output_dir", type=str, default=None)
    combine_parser.add_argument(
        "--axis_bounds",
        nargs=4,
        type=float,
        metavar=("L_X", "H_X", "L_Y", "H_Y"),
        help="axis limits as l_x h_x l_y h_y",
        default=None
    )
    combine_parser.add_argument("--font_size", type=int, default=8)
    combine_parser.add_argument(
        "--guidelines",
        type=lambda x: x.lower() == "true",
        default=True
    )
    combine_parser.add_argument("--opacity", type=float, default=1)
    combine_parser.add_argument("--color_on", type=str, default=None)
    combine_parser.add_argument("--shape_on", type=str, default=None)
    combine_parser.add_argument(
        "--colormap",
        type=str,
        default="okabe_ito",
        help="Name of the matplotlib colormap/color palette to use for point colors "
             "(e.g. okabe_ito, viridis, plasma, tab10)."
    )
    _add_style_args(combine_parser)

    plot_parser = subparsers.add_parser("plot", help="Plot TM-score results from CSV")
    plot_parser.add_argument("--data_file", type=str, required=True)
    plot_parser.add_argument("--save_file", type=str, default=None)
    plot_parser.add_argument("--protein", type=str, default=None)
    plot_parser.add_argument("--title", type=str, default=None)
    plot_parser.add_argument("--model", type=str, default=None)
    plot_parser.add_argument("--seed", type=str, default=None)
    plot_parser.add_argument(
        "--limit_axis",
        type=lambda x: x.lower() == "true",
        default=True
    )
    plot_parser.add_argument("--output_dir", type=str, default=None)
    plot_parser.add_argument(
        "--axis_bounds",
        nargs=4,
        type=float,
        metavar=("L_X", "H_X", "L_Y", "H_Y"),
        help="axis limits as l_x h_x l_y h_y",
        default=None
    )
    plot_parser.add_argument("--font_size", type=int, default=8)
    plot_parser.add_argument(
        "--guidelines",
        type=lambda x: x.lower() == "true",
        default=True
    )
    plot_parser.add_argument("--opacity", type=float, default=1)
    plot_parser.add_argument("--color_on", type=str, default=None)
    plot_parser.add_argument("--shape_on", type=str, default=None)
    plot_parser.add_argument(
        "--colormap",
        type=str,
        default="okabe_ito",
        help="Name of the matplotlib colormap/color palette to use for point colors "
             "(e.g. okabe_ito, viridis, plasma, tab10)."
    )
    _add_style_args(plot_parser)

    full_parser = subparsers.add_parser("all", help="Run calc + plot")
    full_parser.add_argument("--protein", type=str, default=None)
    full_parser.add_argument("--metadata_path", type=str, required=True)
    full_parser.add_argument("--reference_folder", type=str, required=True)
    full_parser.add_argument("--target_folder", type=str, required=True)
    full_parser.add_argument("--output_file", type=str, default=None)
    full_parser.add_argument("--save_file", type=str, default=None)
    full_parser.add_argument("--title", type=str, default=None)
    full_parser.add_argument("--model", type=str, default=None)
    full_parser.add_argument("--seed", type=str, default=None)
    full_parser.add_argument(
        "--limit_axis",
        type=lambda x: x.lower() == "true",
        default=True
    )
    full_parser.add_argument("--output_dir", type=str, default=None)
    full_parser.add_argument(
        "--axis_bounds",
        nargs=4,
        type=float,
        metavar=("L_X", "H_X", "L_Y", "H_Y"),
        help="axis limits as l_x h_x l_y h_y",
        default=None
    )
    full_parser.add_argument("--font_size", type=int, default=8)
    full_parser.add_argument(
        "--guidelines",
        type=lambda x: x.lower() == "true",
        default=True
    )
    full_parser.add_argument("--opacity", type=float, default=1)
    full_parser.add_argument("--color_on", type=str, default=None)
    full_parser.add_argument("--shape_on", type=str, default=None)
    full_parser.add_argument(
        "--colormap",
        type=str,
        default="okabe_ito",
        help="Name of the matplotlib colormap/color palette to use for point colors "
             "(e.g. okabe_ito, viridis, plasma, tab10)."
    )
    _add_style_args(full_parser)

    args = parser.parse_args()
    assert args.limit_axis in [True, False]
    assert args.guidelines in [True, False]

    if args.command == "calc":
        csv_path = calc_tm_score_folders(
            protein=args.protein,
            reference_folder=args.reference_folder,
            target_folder=args.target_folder,
            output_file_name=args.output_file,
            metadata_file=args.metadata_path,
            output_dir=args.output_dir,
            model=args.model,
            seed=args.seed
        )
        console.print(f"  [green]wrote[/] {csv_path}")
    elif args.command == "combine":
        combine_plots(
            data_files=args.data_files,
            save_file_name=args.save_file,
            protein=args.protein,
            title=args.title,
            limit_axis=args.limit_axis,
            output_dir=args.output_dir,
            axis_bounds=args.axis_bounds,
            plot_guidelines=args.guidelines,
            font_size=args.font_size,
            opacity=args.opacity,
            color_on=args.color_on,
            shape_on=args.shape_on,
            colormap=args.colormap,
            **_style_kwargs(args)
        )
        console.print("  [green]combine + plot complete[/]")
    elif args.command == "plot":
        plot_tm_score(
            data_file=args.data_file,
            save_file_name=args.save_file,
            protein=args.protein,
            title=args.title,
            limit_axis=args.limit_axis,
            output_dir=args.output_dir,
            axis_bounds=args.axis_bounds,
            plot_guidelines=args.guidelines,
            font_size=args.font_size,
            opacity=args.opacity,
            color_on=args.color_on,
            shape_on=args.shape_on,
            colormap=args.colormap,
            **_style_kwargs(args)
        )
        console.print("  [green]plot complete[/]")
    elif args.command == "all":
        _ = calc_tm_score_folders(
            protein=args.protein,
            reference_folder=args.reference_folder,
            target_folder=args.target_folder,
            output_file_name=args.output_file,
            metadata_file=args.metadata_path,
            output_dir=args.output_dir,
            model=args.model,
            seed=args.seed
        )
        plot_tm_score(
            data_file=args.output_file,
            save_file_name=args.save_file,
            protein=args.protein,
            title=args.title,
            limit_axis=args.limit_axis,
            output_dir=args.output_dir,
            axis_bounds=args.axis_bounds,
            plot_guidelines=args.guidelines,
            font_size=args.font_size,
            opacity=args.opacity,
            color_on=args.color_on,
            shape_on=args.shape_on,
            colormap=args.colormap,
            **_style_kwargs(args)
        )
        console.print("  [green]calc + plot complete[/]")


if __name__ == "__main__":
    main()