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
        "ASCT2": 0.5,
        "STP10": 0.8,
        "LAT1": 0.85,
        "ZnT8": 0.73,
        "MCT1": 0.70,
        "CCR5": 0.85,
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
        colormap: str = "okabe_ito"
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
    :param colormap: name of the matplotlib colormap to use for coloring
            points (e.g. "okabe_ito", "viridis", "plasma", ...)

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

    cmap_obj = plt.get_cmap(colormap)

    color_is_categorical = color_on != "nseq" and not pd.api.types.is_numeric_dtype(data[color_on])
    if color_is_categorical:
        color_categories = sorted(data[color_on].unique(), key=_natural_sort_key)
        category_color_map = {
            cat: mcolors.to_hex(cmap_obj(i % cmap_obj.N))
            for i, cat in enumerate(color_categories)
        }

    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.grid(True, color="lightgray", linewidth=0.5, alpha=0.4)

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

        ax.legend(
            handles=shape_handles,
            title=f"{shape_on} shape values:",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=len(shape_handles),
            fontsize=font_size + 2,
            title_fontproperties=FontProperties(weight="bold", size=font_size + 3)
        )
    else:
        if color_on == "nseq":
            c_vals = [depth_color(d, colormap) for d in data[color_on]]
            cmap, norm = None, None
        elif color_is_categorical:
            c_vals = [category_color_map[v] for v in data[color_on]]
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

    ax.set_xlabel(
        f"TM-Score: Pred vs {x_label}",
        fontsize=font_size + 1,
        labelpad=font_size + 4,
        fontweight="bold"
    )

    ax.set_ylabel(
        f"TM-Score: Pred vs {y_label}",
        fontsize=font_size + 1,
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

    if color_on == "nseq":
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
                markersize=6,
                label=str(cat),
            )
            for cat in color_categories
        ]

        # A shape legend may already sit at the bottom; keep it and add the
        # color legend on the right so both are visible.
        existing_legend = ax.get_legend()
        color_legend = ax.legend(
            handles=color_handles,
            title=f"{color_on}:",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=font_size + 1,
            title_fontproperties=FontProperties(weight="bold", size=font_size + 2),
        )
        if existing_legend is not None:
            ax.add_artist(existing_legend)
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

    _round_ticks_with_one(ax, "x")
    _round_ticks_with_one(ax, "y")

    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

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

    plt.savefig(
        save_file_name,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


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
        colormap: str = "okabe_ito"
) -> None:
    """
    :param colormap: name of the matplotlib colormap to use for coloring
            points (e.g. "okabe_ito", "viridis", "plasma", ...), forwarded
            to plot_tm_score for the combined plot.
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
        filename,
        save_file_name,
        protein,
        title,
        limit_axis,
        output_dir,
        experiment_name,
        axis_bounds,
        plot_guidelines,
        font_size,
        opacity,
        color_on,
        shape_on,
        colormap
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
            colormap=args.colormap
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
            colormap=args.colormap
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
            colormap=args.colormap
        )
        console.print("  [green]calc + plot complete[/]")


if __name__ == "__main__":
    main()