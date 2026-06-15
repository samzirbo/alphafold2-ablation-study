from pathlib import Path
import warnings

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

from scripts.plot_tmscore import get_reference_files

console = Console()


def plot_pca(
    protein: str,
    reference_folder: str,
    target_folder: str,
    metadata_file: str,
    output_file_name: str = None,
    output_dir: str = ".",
    model: str = None,
    seed: str = None,
):
    reference_folder = reference_folder + "/" if reference_folder[-1] != "/" else reference_folder
    target_folder = target_folder + "/" if target_folder[-1] != "/" else target_folder

    parser = PDBParser(QUIET=True)

    reference_files = get_reference_files(protein, reference_folder, metadata_file)

    if len(reference_files) == 0:
        raise ValueError("No reference structures found")

    target_files = [
        str(f)
        for f in Path(target_folder).glob(
            f"{protein}_unrelaxed_*_alphafold2_model_{'*' if model is None else model}_seed_{'*' if seed is None else seed}.pdb"
        )
    ]

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
            labels = []

            task = progress.add_task("Processing", total=len(target_files))

            for target_file in target_files:
                structure = parser.get_structure("model",target_file)
                model_ca = {
                    residue.id[1]: residue["CA"]
                    for residue in structure.get_residues()
                    if "CA" in residue
                }

                # residues present in BOTH structures
                common_res = sorted(set(ref_ca.keys()) &set(model_ca.keys()))
                ref_atoms = [
                    ref_ca[r]
                    for r in common_res
                ]
                model_atoms = [
                    model_ca[r]
                    for r in common_res
                ]

                sup = Superimposer()
                sup.set_atoms(ref_atoms, model_atoms)
                sup.apply(structure.get_atoms())

                xyz = np.array(
                    [
                        model_ca[r].coord
                        for r in common_res
                    ]
                ).flatten()

                coords.append(xyz)
                labels.append(
                    Path(target_file).stem
                )

                progress.advance(task)

            if len(coords) < 3:
                raise ValueError(
                    "Not enough structures for PCA"
                )

            X = np.array(coords)
            pca = PCA(n_components=2)
            proj = pca.fit_transform(X)

            plt.figure(figsize=(8, 6), dpi=300)

            plt.scatter(
                proj[:, 0],
                proj[:, 1],
                s=80,
                color="orange"
            )
            plt.xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
            plt.ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
            plt.title(f"{protein}")
            plt.tight_layout()
            if output_file_name:
                output_path = (Path(output_dir)/ f"{output_file_name + '_ref_' + reference_file.split("/")[-1]}.png")
                plt.savefig(
                    output_path,
                    dpi=300,
                    bbox_inches="tight",
                )
                print(f"Saved PCA plot to {output_path}")