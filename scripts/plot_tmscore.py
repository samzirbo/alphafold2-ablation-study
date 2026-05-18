import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt
from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data


def __calc_tm_score(file1_path: str, file2_path: str) -> float:
    """
    :param file1_path: path to .pdb file
    :param file2_path: path to .pdb file

    Calculate TM-score between proteins in file1 and file2.
    """
    s1 = get_structure(file1_path)
    s2 = get_structure(file2_path)

    coords1, seq1 = get_residue_data(next(s1.get_chains()))
    coords2, seq2 = get_residue_data(next(s2.get_chains()))

    tm_score = tm_align(coords1, coords2, seq1, seq2).tm_norm_chain1
    return tm_score


def calc_tm_score_folders(
        protein: str,
        reference_folder: str,
        target_folder: str,
        metadata_file: str,
        output_file_name: str
) -> None:
    """
    :param protein: protein name
    :param reference_folder: path to folder containing reference files
    :param target_folder: path to folder containing target files
    :param metadata_file: path to metadata file
    :param output_file_name: name of output csv file

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

    metadata = json.load(open(metadata_file))

    IF_ID = metadata[protein]["conformations"]["state_1"]["pdb_id"]
    IF_CHAIN = metadata[protein]["conformations"]["state_1"]["chain"]
    IF_LABEL = metadata[protein]["conformations"]["state_1"]["label"]
    OF_ID = metadata[protein]["conformations"]["state_2"]["pdb_id"]
    OF_CHAIN = metadata[protein]["conformations"]["state_2"]["chain"]
    OF_LABEL = metadata[protein]["conformations"]["state_2"]["label"]

    reference_files = [
        reference_folder + f"{IF_LABEL}_{IF_ID}_{IF_CHAIN}.pdb",
        reference_folder + f"{OF_LABEL}_{OF_ID}_{OF_CHAIN}.pdb"
    ]

    target_files = [str(f) for f in Path(target_folder).glob(f"{protein}_unrelaxed_*_alphafold2_model_*_seed_*.pdb")]

    assert len(reference_files) == 2
    assert len(target_files) == 25

    results_df = pd.DataFrame()

    for target_file in target_files:
        row = {"protein": protein, "nseq": nseq}
        for reference_file in reference_files:
            tm_score = __calc_tm_score(
                reference_file,
                target_file
            )
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
                raise f"File {reference_file} not recognized"
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

    results_df.to_csv(output_file_name, index=False)


def plot_tm_score(data_file: str, save_file_name: str = None, protein: str = None) -> None:
    """
    :param data_file: path to csv file
    :param save_file_name: name of output image file
    :param protein: protein name if more than 1 protein in csv

    Scatterplot of IF-OF / inactive-active TM-scores for different MSA depths.
    """
    data = pd.read_csv(data_file)

    if protein is None:
        assert data["protein"].unique().size == 1
    else:
        data = data[data["protein"] == protein]

    assert ("tm_IF" in data.columns and "tm_OF" in data.columns) or ("tm_A" in data.columns and "tm_I" in data.columns)
    structure_type = "conformational" if "tm_IF" in data.columns else "functional"

    if structure_type == "conformational":
        x_col_name = "tm_IF"
        y_col_name = "tm_OF"
    else:
        x_col_name = "tm_I"
        y_col_name = "tm_A"

    data = data.sort_values("nseq")

    base_cmap = plt.get_cmap("cividis")

    new_colors = base_cmap(np.linspace(0.3, 1.0, 256))
    cmap = colors.LinearSegmentedColormap.from_list("truncated_cividis", new_colors)

    norm = colors.LogNorm(
        vmin=data["nseq"].min(),
        vmax=data["nseq"].max()
    )

    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.grid(True, color="lightgray", linewidth=0.5, alpha=0.4)

    scatter = ax.scatter(
        data[x_col_name],
        data[y_col_name],
        c=data["nseq"],
        cmap=cmap,
        norm=norm,
        s=25,
        edgecolors="black",
        linewidths=0.375
    )

    x_label = "inward-facing" if structure_type == "conformational" else "inactive"
    y_label = "outward-facing" if structure_type == "conformational" else "active"

    ax.set_xlabel(
        f"Similarity to {x_label} conformation (TM-score)",
        fontsize=7,
        fontweight="bold"
    )

    ax.set_ylabel(
        f"Similarity to {y_label} conformation (TM-score)",
        fontsize=7,
        fontweight="bold"
    )

    ax.tick_params(axis="both", labelsize=6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("black")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("MSA depth (sequences)", fontsize=7, fontweight="bold")
    cbar.ax.tick_params(labelsize=6)

    ticks = np.sort(data["nseq"].unique())
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(int(t)) for t in ticks])

    cbar.ax.minorticks_off()
    cbar.ax.tick_params(which="minor", length=0)

    ax.set_aspect("auto", adjustable="box")

    if save_file_name is not None:
        plt.savefig(
            save_file_name,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    calc_parser = subparsers.add_parser("calc", help="Calculate TM-scores and save CSV")
    calc_parser.add_argument("--protein", type=str, required=True)
    calc_parser.add_argument("--reference_folder", type=str, required=True)
    calc_parser.add_argument("--target_folder", type=str, required=True)
    calc_parser.add_argument("--output_file", type=str, required=True)
    calc_parser.add_argument("--metadata_path", type=str, required=True)

    plot_parser = subparsers.add_parser("plot", help="Plot TM-score results from CSV")
    plot_parser.add_argument("--data_file", type=str, required=True)
    plot_parser.add_argument("--save_file", type=str, default=None)
    plot_parser.add_argument("--protein", type=str, default=None)

    full_parser = subparsers.add_parser("all", help="Run calc + plot")
    full_parser.add_argument("--protein", type=str, default=None)
    full_parser.add_argument("--metadata_path", type=str, required=True)
    full_parser.add_argument("--reference_folder", type=str, required=True)
    full_parser.add_argument("--target_folder", type=str, required=True)
    full_parser.add_argument("--output_file", type=str, required=True)
    full_parser.add_argument("--save_file", type=str, default=None)

    args = parser.parse_args()
    if args.command == "calc":
        calc_tm_score_folders(
            protein=args.protein,
            reference_folder=args.reference_folder,
            target_folder=args.target_folder,
            output_file_name=args.output_file,
            metadata_file=args.metadata_path,
        )
    elif args.command == "plot":
        plot_tm_score(
            data_file=args.data_file,
            save_file_name=args.save_file,
            protein=args.protein,
        )
    elif args.command == "all":
        calc_tm_score_folders(
            protein=args.protein,
            reference_folder=args.reference_folder,
            target_folder=args.target_folder,
            output_file_name=args.output_file,
            metadata_file=args.metadata_path
        )
        plot_tm_score(
            data_file=args.output_file,
            save_file_name=args.save_file,
            protein=args.protein,
        )


if __name__ == "__main__":
    main()
