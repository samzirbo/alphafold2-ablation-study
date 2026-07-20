import argparse
import json
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
import sys


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
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {protein}: Calculating TM-scores for {len(target_files)} models...", flush=True)

    for i, target_file in enumerate(target_files):
        if i > 0 and i % 5 == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {protein}: Processed {i}/{len(target_files)} models...", flush=True)
        row = {
            "protein": protein,
            "nseq": nseq,
            "reference_tm": reference_tm,
            "seed": int(target_file.split("_")[-1].split(".")[0]),
            "model": int(target_file.split("_")[-3]),
            "experiment": output_dir.split("/")[-1]
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
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {protein}: Finished TM-score calculations.", flush=True)

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
        "PTH1R": 0.82,
        "ASCT2": 0.6,
        "STP10": 0.8,
        "LAT1": 0.85,
        "ZnT8": 0.73,
        "MCT1": 0.75,
        "CCR5": 0.7,
        "MurJ": 0.7,
        "PfMATE": 0.7,
        "SERT": 0.7
    }[protein]


def depth_color(depth: int):
    values = [16, 32, 64, 128, 256, 512, 1024, 5120]
    cmap = plt.cm.okabe_ito
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
    cmap = plt.cm.okabe_ito
    return [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]


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
        colors: list = None
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

    data = data.sort_values(color_on)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.grid(True, color="lightgray", linewidth=0.5, alpha=0.4)

    color_is_numeric = pd.api.types.is_numeric_dtype(data[color_on])
    categorical_color = (color_on != "nseq") and not color_is_numeric

    assert legend_layout in {"auto", "row", "column"}, \
        f"legend_layout must be 'auto', 'row', or 'column', got {legend_layout!r}"

    # Resolve absolute font sizes; fall back to the historical font_size + N defaults.
    axis_title_size = axis_title_fontsize if axis_title_fontsize is not None else font_size + 1
    legend_title_size = legend_title_fontsize if legend_title_fontsize is not None else font_size + 3
    legend_text_size = legend_text_fontsize if legend_text_fontsize is not None else font_size + 2

    if shape_on is not None:
        markers = [
            "o", "s", "^", "D", "v", "<", ">", "P", "X",
            "*", "h", "H", "8", "p", "d"
        ]

        categories = data[shape_on].dropna().unique()
        marker_map = {cat: markers[i % len(markers)] for i, cat in enumerate(categories)}

        for cat in categories:
            mask = data[shape_on] == cat
            scatter = ax.scatter(
                data.loc[mask, x_col_name],
                data.loc[mask, y_col_name],
                c=[depth_color(d) for d in data.loc[mask, color_on]] if color_on == "nseq" else data.loc[mask, color_on],
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
            ncol=1 if legend_layout == "column" else len(shape_handles),
            fontsize=legend_text_size,
            title_fontproperties=FontProperties(weight="bold", size=legend_title_size)
        )
    elif categorical_color:
        categories = np.sort(data[color_on].dropna().unique())
        palette = colors if colors is not None else get_okabe_ito_colors()
        color_map = {
            cat: palette[i % len(palette)] for i, cat in enumerate(categories)
        }
        for cat in categories:
            mask = data[color_on] == cat
            ax.scatter(
                data.loc[mask, x_col_name], data.loc[mask, y_col_name],
                c=color_map[cat], s=30, edgecolors="black",
                linewidths=0.375, alpha=opacity,
            )
        color_handles = [
            Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=color_map[cat], markeredgecolor="black",
                   markersize=6,
                   label=(legend_labels[i] if legend_labels is not None
                          and i < len(legend_labels) else str(cat)))
            for i, cat in enumerate(categories)
        ]
        ax.legend(
            handles=color_handles,
            title=legend_title if legend_title is not None else f"{color_on}:",
            loc="upper center", bbox_to_anchor=(0.5, -0.15),
            ncol=1 if legend_layout == "column" else len(color_handles),
            fontsize=legend_text_size,
            title_fontproperties=FontProperties(weight="bold", size=legend_title_size),
        )


    else:
        if color_on == "nseq":
            c_vals = [depth_color(d) for d in data[color_on]]
            cmap, norm = None, None
        else:
            unique_vals = np.sort(data[color_on].unique())
            c_vals = [np.where(unique_vals == v)[0][0] for v in data[color_on]]
            cmap = "okabe_ito"
            norm = mcolors.BoundaryNorm(np.arange(-0.5, len(unique_vals) + 0.5), plt.cm.okabe_ito.N)

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

    x_label = "inward-facing" if structure_type == "conformational" else "inactive"
    y_label = "outward-facing" if structure_type == "conformational" else "active"

    xlabel = x_axis_title if x_axis_title is not None else f"Similarity to {x_label} conformation (TM-score)"
    ylabel = y_axis_title if y_axis_title is not None else f"Similarity to {y_label} conformation (TM-score)"

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

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax.tick_params(axis="both", labelsize=font_size + 3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")

    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("black")

    if plot_guidelines:
        reference_tm = float(data.loc[0]["reference_tm"])
        plt.axvline(x=reference_tm, c="gray", linestyle="--")
        plt.axhline(y=reference_tm, c="gray", linestyle="--")

    unique_depths = np.sort(data[color_on].unique())

    if color_on == "nseq":
        if len(unique_depths) > 1:
            depth_text = ", ".join([str(x) for x in unique_depths])

            used_colors = [depth_color(d) for d in unique_depths]
            cmap = mcolors.ListedColormap(used_colors)

            bounds = np.arange(len(unique_depths) + 1)
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
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
    else:
        depth_text = f"{', '.join([str(x) for x in np.sort(data['nseq'].unique()).tolist()])}"
        if not categorical_color:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(
                color_on,
                fontsize=font_size + 1,
                fontweight="bold",
                labelpad=font_size + 1
            )
            unique_vals = np.sort(data[color_on].unique())
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

    # Custom tick grid aligned to `tick_anchor` with spacing `tick_size`, filling the whole
    # (already-limited) axis range on both x and y. Overrides the MaxNLocator default above.
    if tick_anchor is not None and tick_size is not None:
        for set_ticks, lim in [(ax.set_xticks, ax.get_xlim()), (ax.set_yticks, ax.get_ylim())]:
            low, high = min(lim), max(lim)
            k_min = int(np.ceil((low - tick_anchor) / tick_size))
            k_max = int(np.floor((high - tick_anchor) / tick_size))
            ticks = np.round(tick_anchor + tick_size * np.arange(k_min, k_max + 1), 8)
            set_ticks(ticks)
        # set_xticks/set_yticks replace the label objects, so re-apply size + bold weight.
        ax.tick_params(axis="both", labelsize=font_size + 3)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontweight("bold")

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
        colors: list = None
) -> None:
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

    filename = (
        f"{('').join(data_files[0].split('.')[:-1])}_COMBINED_"
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
    elif args.command == "combine":
        combine_plots(
            data_files=args.data_files,
            save_file_name=args.save_file,
            protein=args.protein,
            title=args.title,
            limit_axis=args.limit_axis,
            output_dir=args.output_dir,
            axis_bounds=args.axis_bounds,
            plot_guidelines=args.plot_guidelines,
            font_size=args.font_size,
            opacity=args.opacity,
            color_on=args.color_on,
            shape_on=args.shape_on
        )
        print(f"  wrote {csv_path}")
    elif args.command == "plot":
        plot_tm_score(
            data_file=args.data_file,
            save_file_name=args.save_file,
            protein=args.protein,
            title=args.title,
            limit_axis=args.limit_axis,
            output_dir=args.output_dir,
            axis_bounds=args.axis_bounds,
            plot_guidelines=args.plot_guidelines,
            font_size=args.font_size,
            opacity=args.opacity,
            color_on=args.color_on,
            shape_on=args.shape_on
        )
        print("  plot complete")
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
            data_file=args.data_file,
            save_file_name=args.save_file,
            protein=args.protein,
            title=args.title,
            limit_axis=args.limit_axis,
            output_dir=args.output_dir,
            axis_bounds=args.axis_bounds,
            plot_guidelines=args.plot_guidelines,
            font_size=args.font_size,
            opacity=args.opacity,
            color_on=args.color_on,
            shape_on=args.shape_on
        )
        print("  calc + plot complete")


if __name__ == "__main__":
    main()
