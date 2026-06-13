import argparse
import glob
import os

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


def autoplot_tmscore(
        result_path,
        base_repo_path,
        limit_axis=True,
        experiment_folder_name=None,
        force_replot=False,
        axis_bounds=None,
        plot_guidelines=True,
        font_size=8,
        opacity=1,
        model=None,
        seed=None,
        color_on="nseq",
        shape_on=None
):
    if experiment_folder_name is None:
        print("Plotting all subfolders of ", result_path)
    else:
        print("Plotting only experiment ", result_path + experiment_folder_name)

    all_subdirectories = glob.glob(os.path.join(result_path, '*'))

    filtered_subdirectories = []
    for subdir in all_subdirectories:
        if experiment_folder_name is None:
            if os.path.isdir(subdir) and os.path.basename(subdir) not in directories_to_exclude:
                filtered_subdirectories.append(subdir)
        else:
            if os.path.isdir(subdir) and experiment_folder_name == os.path.basename(subdir):
                filtered_subdirectories.append(subdir)
                break
    if not filtered_subdirectories:
        print("No experiments found in ", result_path)

    plots_dir = os.path.join(result_path, "plots/TM_Score")
    os.makedirs(plots_dir, exist_ok=True)
    print("Saving results in ", plots_dir)

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
                                        experiment_result_dir,
                                        model,
                                        seed
                                    )
                                    print("Generated TM-score CSV file: " + csv_path)
                                plot_tm_score(
                                    experiment_result_dir + "/" + protein_name + ".csv",
                                    save_file_name=protein_name + ".png",
                                    protein=protein_name,
                                    limit_axis=limit_axis,
                                    output_dir=experiment_result_dir,
                                    experiment_name=experiment_name,
                                    axis_bounds=axis_bounds,
                                    plot_guidelines=plot_guidelines,
                                    font_size=font_size,
                                    opacity=opacity,
                                    color_on=color_on,
                                    shape_on=shape_on,
                                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_path", required=True)
    parser.add_argument("--base_repo_path", required=True)
    parser.add_argument("--experiment_folder_name", required=False)
    parser.add_argument(
        "--limit_axis",
        type=lambda x: x.lower() == "true",
        default=True
    )
    parser.add_argument(
        "--replot_all",
        type=lambda x: x.lower() == "true",
        default=False
    )
    parser.add_argument(
        "--axis_bounds",
        nargs=4,
        type=float,
        metavar=("L_X", "H_X", "L_Y", "H_Y"),
        help="axis limits as l_x h_x l_y h_y",
        default=None
    )
    parser.add_argument("--font_size", type=int, default=8)
    parser.add_argument(
        "--guidelines",
        type=lambda x: x.lower() == "true",
        default=True
    )
    parser.add_argument("--opacity", type=float, default=1)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--color_on", type=str)
    parser.add_argument("--shape_on", type=str, default=None)

    args = parser.parse_args()

    autoplot_tmscore(
        args.result_path,
        args.base_repo_path,
        args.limit_axis,
        args.experiment_folder_name,
        args.replot_all,
        args.axis_bounds,
        args.guidelines,
        args.font_size,
        args.opacity,
        args.model,
        args.seed,
        args.color_on,
        args.shape_on
    )
