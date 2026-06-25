from pathlib import Path
import warnings

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from Bio.PDB import PDBParser, Superimposer
from sklearn.decomposition import PCA

from rich.progress import (
    SpinnerColumn,
    Progress,
    BarColumn,
    TextColumn,
    MofNCompleteColumn,
)
from rich.console import Console

from scripts.plot_tmscore import get_reference_files, __calc_tm_score

console = Console()


def plot_pca(
    protein: str,
    reference_folder: str,
    target_folder: str,
    metadata_file: str,
    experiment_name: str,
    output_file_name: str = None,
    output_dir: str = ".",
    font_size= 12
):
    reference_folder = reference_folder + "/" if reference_folder[-1] != "/" else reference_folder
    target_folder = target_folder + "/" if target_folder[-1] != "/" else target_folder

    parser = PDBParser(QUIET=True)

    reference_files = get_reference_files(protein, reference_folder, metadata_file)

    if len(reference_files) == 0:
        raise ValueError("No reference structures found")

    target_files = [str(f) for f in Path(target_folder).glob(f"{protein}_unrelaxed_*_alphafold2_model_*_seed_*.pdb")]

    if len(target_files) != 25:
        warnings.warn(
            f"There are {len(target_files)} models instead of 25"
        )

    progress = Progress(
        SpinnerColumn(),
        TextColumn(f"{protein} PCA"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    )

    with progress:
        for reference_file in reference_files:
            ref_structure = parser.get_structure("reference", reference_file)
            ref_ca = {
                residue.id[1]: residue["CA"]
                for residue in ref_structure.get_residues()
                if "CA" in residue
            }

            coords = []
            models = []
            tm_scores = []

            task = progress.add_task("Processing", total=len(target_files))

            for target_file in target_files:
                structure = parser.get_structure("model", target_file)

                model_ca = {
                    residue.id[1]: residue["CA"]
                    for residue in structure.get_residues()
                    if "CA" in residue
                }

                common_res = sorted(set(ref_ca.keys()) & set(model_ca.keys()))

                ref_atoms = [ref_ca[r] for r in common_res]
                model_atoms = [model_ca[r] for r in common_res]

                sup = Superimposer()
                sup.set_atoms(ref_atoms, model_atoms)
                sup.apply(structure.get_atoms())

                xyz = np.array([model_ca[r].coord for r in common_res]).flatten()

                coords.append(xyz)
                models.append(int(Path(target_file).stem.split("_")[-3]))

                tm_score, _, _ = __calc_tm_score(reference_file, target_file)
                tm_scores.append(tm_score)

                progress.advance(task)

            if len(coords) < 3:
                raise ValueError("Not enough structures for PCA")

            X = np.array(coords)
            pca = PCA(n_components=2)
            proj = pca.fit_transform(X)

            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

            norm = matplotlib.colors.Normalize(
                vmin=min(tm_scores),
                vmax=max(tm_scores),
            )
            cmap = plt.get_cmap("plasma")

            unique_models = sorted(set(models))
            markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

            model_marker = {
                m: markers[i % len(markers)]
                for i, m in enumerate(unique_models)
            }

            for x, y, model, tm in zip(proj[:, 0], proj[:, 1], models, tm_scores):
                ax.scatter(
                    x,
                    y,
                    s=80,
                    c=[cmap(norm(tm))],
                    marker=model_marker[model],
                    edgecolor="black",
                    linewidth=0.5,
                )

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("TM-score", fontweight="bold", fontsize=font_size)
            cbar.ax.tick_params(labelsize=font_size)

            model_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker=model_marker[m],
                    linestyle="None",
                    color="black",
                    markerfacecolor="gray",
                    markeredgecolor="black",
                    markersize=font_size * 0.5,
                    label=f"Model {m}",
                )
                for m in unique_models
            ]

            legend = ax.legend(
                handles=model_handles,
                title="Models",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.12),
                ncol=min(len(model_handles), 6),
                prop={"size": font_size, "weight": "bold"},
            )

            legend.set_title("Models", prop={"size": font_size, "weight": "bold"})

            fig.subplots_adjust(bottom=0.25)

            ax.set_xlabel(
                f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)",
                fontsize=font_size,
                fontweight="bold",
            )
            ax.set_ylabel(
                f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)",
                fontsize=font_size,
                fontweight="bold",
            )
            ax.set_title(
                experiment_name + " | " + protein,
                fontsize=font_size + 3,
                fontweight="bold",
            )

            ax.tick_params(axis="both", labelsize=font_size)

            if output_file_name:
                output_path = Path(output_dir) / f"{output_file_name}_ref_{reference_file.split('/')[-1]}.png"
            else:
                output_path = Path(output_dir) / f"{reference_file.split('/')[-1]}.png"

            fig.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)