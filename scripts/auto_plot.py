import os
import glob
import argparse

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

import plot_tmscore

console = Console(highlight=False)

directories_to_exclude = ["archive", "plots"]
rerun_choices = ("missing", "plots", "csv", "all")

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
        rerun="missing"
):
    if rerun not in rerun_choices:
        raise ValueError(f"rerun must be one of {rerun_choices}, got {rerun!r}")

    if experiment_folder_name is None:
        console.print("[bold]Plotting all subfolders[/]")
    else:
        console.print(f"[bold]Plotting only experiment[/] {experiment_folder_name}")
    console.print(f"  results: [dim]{result_path}[/]")
    console.print(f"  rerun:   [cyan]{rerun}[/]")

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
        console.print(f"  [yellow]warning[/] No experiments found in {result_path}")
        return

    plots_dir = os.path.join(result_path, "plots/TM_Score")
    os.makedirs(plots_dir, exist_ok=True)

    jobs = []
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
                            jobs.append((experiment_name, protein_name, full_path))

    if not jobs:
        console.print("  [yellow]warning[/] No protein prediction folders found.")
        return

    stats = {"csv": 0, "plots": 0, "skipped": 0, "failed": 0}
    failures = []
    experiments = []
    for experiment_name, _, _ in jobs:
        if experiment_name not in experiments:
            experiments.append(experiment_name)
    experiment_numbers = {name: i + 1 for i, name in enumerate(experiments)}

    for experiment_name in experiments:
        experiment_jobs = [job for job in jobs if job[0] == experiment_name]
        experiment_stats = {"csv": 0, "plots": 0, "skipped": 0, "failed": 0}

        console.print(
            f"\n[bold cyan]Experiment {experiment_numbers[experiment_name]}/{len(experiments)}:[/] "
            f"{experiment_name}"
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("  {task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("proteins"),
            console=console,
            transient=True,
        )

        with progress:
            task = progress.add_task("proteins", total=len(experiment_jobs))

            for _, protein_name, full_path in experiment_jobs:
                progress.update(task, description=protein_name)

                experiment_result_dir = plots_dir + "/" + experiment_name
                csv_path = experiment_result_dir + "/" + protein_name + ".csv"
                png_path = experiment_result_dir + "/" + protein_name + ".png"
                make_csv = rerun in {"csv", "all"} or not os.path.exists(csv_path)
                make_plot = rerun in {"plots", "all"} or not os.path.exists(png_path)

                if not make_csv and not make_plot:
                    stats["skipped"] += 1
                    experiment_stats["skipped"] += 1
                    progress.advance(task)
                    continue

                try:
                    os.makedirs(experiment_result_dir, exist_ok=True)

                    if make_csv:
                        previous_console = plot_tmscore.console
                        with open(os.devnull, "w") as devnull:
                            try:
                                plot_tmscore.console = Console(
                                    file=devnull,
                                    force_terminal=False,
                                    color_system=None,
                                )
                                plot_tmscore.calc_tm_score_folders(
                                    protein_name,
                                    base_repo_path + "/data/" + protein_name + "/references/",
                                    full_path,
                                    base_repo_path + "/data/metadata.json",
                                    protein_name + ".csv",
                                    experiment_result_dir,
                                )
                            finally:
                                plot_tmscore.console = previous_console
                        stats["csv"] += 1
                        experiment_stats["csv"] += 1

                    if make_plot:
                        plot_tmscore.plot_tm_score(
                            experiment_result_dir + "/" + protein_name + ".csv",
                            protein=protein_name,
                            save_file_name=protein_name + ".png",
                            limit_axis=limit_axis,
                            output_dir=experiment_result_dir,
                            experiment_name=experiment_name
                        )
                        stats["plots"] += 1
                        experiment_stats["plots"] += 1
                except Exception as e:
                    stats["failed"] += 1
                    experiment_stats["failed"] += 1
                    failures.append(f"{experiment_name}/{protein_name}: {e}")
                finally:
                    progress.advance(task)

        console.print(
            f"  [green]done[/] {experiment_name}: "
            f"CSV {experiment_stats['csv']}, plots {experiment_stats['plots']}, "
            f"skipped {experiment_stats['skipped']}, failed {experiment_stats['failed']}"
        )

    console.print(
        f"\n[bold green]Autoplot complete.[/] "
        f"CSV: {stats['csv']}, plots: {stats['plots']}, "
        f"skipped: {stats['skipped']}, failed: {stats['failed']}\n"
    )

    if failures:
        console.print("[red bold]Failures:[/]")
        for failure in failures:
            console.print(f"  [red]-[/] {failure}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_path", required=True)
    parser.add_argument("--base_repo_path", required=True)
    parser.add_argument("--experiment_folder_name", required=False)
    parser.add_argument(
        "--limit_axis",
        action=argparse.BooleanOptionalAction,
        default=True
    )
    parser.add_argument(
        "--rerun",
        choices=rerun_choices,
        default="missing",
        help="Regenerate missing outputs only, plots only, CSVs only, or CSVs and plots."
    )

    args = parser.parse_args()

    autoplot_tmscore(
        args.result_path,
        args.base_repo_path,
        args.limit_axis,
        args.experiment_folder_name,
        args.rerun
    )
