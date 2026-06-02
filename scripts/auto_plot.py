import os
import glob
import sys
import argparse

from plot_tmscore import calc_tm_score_folders, plot_tm_score

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


def autoplot(result_path, base_repo_path, limit_axis=False):
    print(result_path, base_repo_path)
    all_subdirectories = glob.glob(os.path.join(result_path, '*'))

    print(all_subdirectories)

    filtered_subdirectories = []
    for subdir in all_subdirectories:
        if os.path.isdir(subdir) and os.path.basename(subdir) not in directories_to_exclude:
            filtered_subdirectories.append(subdir)

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
                            if len(glob.glob(experiment_result_dir + "/*")) == 0:
                                os.makedirs(experiment_result_dir, exist_ok=True)
                                calc_tm_score_folders(
                                    protein_name,
                                    base_repo_path + "/data/" + protein_name + "/references/",
                                    full_path,
                                    base_repo_path + "/data/metadata.json",
                                    protein_name + ".csv",
                                    experiment_result_dir
                                )
                                plot_tm_score(
                                    experiment_result_dir + "/" + protein_name + ".csv",
                                    protein=protein_name,
                                    save_file_name=protein_name + ".png",
                                    limit_axis=limit_axis,
                                    output_dir=experiment_result_dir
                                )
                            else:
                                print("Skipped: ", experiment_result_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_path", required=True)
    parser.add_argument("--base_repo_path", required=True)
    parser.add_argument("--limit_axis", default=False)

    args = parser.parse_args()

    autoplot(args.result_path, args.base_repo_path)
