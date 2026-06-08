import os
import glob
import sys
import argparse

from plot_tmscore import calc_tm_score_folders, plot_tm_score
from plot_pLDDT import generate_notebook

directories_to_exclude = ["archive", "plots"]

proteins = [
    "PfMATE",
    "MCT1",
    "MurJ",
    "SERT",
    "ASCT2",
    "PTH1R",
    "STP10",
    "CGRPR",
    "LAT1",
    "ZnT8",
    "CCR5",
    "FZD7"
]


def autoplot_tmscore(result_path, base_repo_path, limit_axis=True, experiment_folder_name=None, force_replot=False):
    if experiment_folder_name is None:
        print("Plotting all subfolders of ", base_repo_path)
    else:
        print("Plotting only experiment ", experiment_folder_name)
    print("Saving results in ", result_path)

    all_subdirectories = glob.glob(os.path.join(result_path, '*'))

    filtered_subdirectories = []
    for subdir in all_subdirectories:
        if experiment_folder_name is None:
            if os.path.isdir(subdir) and os.path.basename(subdir) not in directories_to_exclude:
                filtered_subdirectories.append(subdir)
        else:
            if os.path.isdir(subdir) and experiment_folder_name in subdir:
                filtered_subdirectories.append(subdir)
                break
    if not filtered_subdirectories:
        print("No experiments found in ", base_repo_path)

    plots_dir = os.path.join(result_path, "plots/TM_Score")
    os.makedirs(plots_dir, exist_ok=True)

    for start_path in filtered_subdirectories:
        found = []
        for root, dirs, files in os.walk(start_path):
            for protein_name in proteins:
                for d in dirs:
                    if protein_name == d:
                        full_path = os.path.join(root, d)
                        if full_path not in found:
                            found.append(full_path)
                            experiment_name = full_path.split("/")[-2]
                            experiment_result_dir = plots_dir + "/" + experiment_name
                            csv_path = experiment_result_dir + "/" + protein_name + ".csv"
                            png_path = experiment_result_dir + "/" + protein_name + ".png"
                            if os.path.exists(png_path) and not force_replot:
                                print("Skipped: " + experiment_result_dir + "/" + protein_name)
                            else:
                                os.makedirs(experiment_result_dir, exist_ok=True)
                                if not os.path.exists(csv_path):
                                    calc_tm_score_folders(
                                        protein_name,
                                        base_repo_path + "/data/" + protein_name + "/references/",
                                        full_path,
                                        base_repo_path + "/data/metadata.json",
                                        protein_name + ".csv",
                                        experiment_result_dir
                                    )
                                    print("Generated TM-score CSV file: " + csv_path)
                                plot_tm_score(
                                    experiment_result_dir + "/" + protein_name + ".csv",
                                    protein=protein_name,
                                    save_file_name=protein_name + ".png",
                                    limit_axis=limit_axis,
                                    output_dir=experiment_result_dir,
                                    experiment_name=experiment_name
                                )


def autoplot_plddt(base_repo_path):
    print("Reading from ", base_repo_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_path", required=True)
    parser.add_argument("--base_repo_path", required=True)
    parser.add_argument("--experiment_folder_name", required=False)
    parser.add_argument("--limit_axis", default=True)
    parser.add_argument("--replot_all", default=False)
    parser.add_argument("--type", default=None)

    args = parser.parse_args()

    assert args.type in [None, "tmscore", "plddt"]

    if args.type is not "plddt":
        autoplot_tmscore(
            args.result_path,
            args.base_repo_path,
            args.limit_axis,
            args.experiment_folder_name,
            args.replot_all
        )

    if args.type is not "tmscore":
        autoplot_plddt(args.base_repo_path)
