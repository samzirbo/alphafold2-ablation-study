import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

import os
import sys

# --- HARDCODED BASELINE VALUES of Depth 5120 INTEGRATION ---
BASELINE_LOOKUP = {
    "ASCT2":  {"s1": 0.906875, "s2": 0.953802},
    "CCR5":   {"s1": 0.978090, "s2": 0.927801},
    "CGRPR":  {"s1": 0.919797, "s2": 0.949846},
    "FZD7":   {"s1": 0.928276, "s2": 0.937535},
    "LAT1":   {"s1": 0.956259, "s2": 0.959860},
    "MCT1":   {"s1": 0.979506, "s2": 0.817855},
    "MurJ":   {"s1": 0.970554, "s2": 0.749366},
    "PTH1R":  {"s1": 0.876065, "s2": 0.947840},
    "PfMATE": {"s1": 0.745133, "s2": 0.994258},
    "SERT":   {"s1": 0.921699, "s2": 0.988799},
    "STP10":  {"s1": 0.947572, "s2": 0.955642},
    "ZnT8":   {"s1": 0.898871, "s2": 0.896885}
}


TOP_5_BASELINE = {
    "ASCT2":  {"s1": 0.900909, "s2": 0.952056},
    "CCR5":  {"s1": 0.977807, "s2": 0.927206},
    "CGRPR":  {"s1": 0.903961, "s2": 0.948467},
    "FZD7":  {"s1": 0.927170, "s2": 0.937161},
    "LAT1":  {"s1": 0.954654, "s2": 0.933330},
    "MCT1":  {"s1": 0.978847, "s2": 0.815559},
    "MurJ":  {"s1": 0.969450, "s2": 0.747798},
    "PTH1R":  {"s1": 0.872883, "s2": 0.946925},
    "PfMATE":  {"s1": 0.740347, "s2": 0.994068},
    "SERT":  {"s1": 0.919914, "s2": 0.988468},
    "STP10":  {"s1": 0.944204, "s2": 0.951962},
    "ZnT8":  {"s1": 0.896208, "s2": 0.893918},
}


MEAN_LOOKUP = {
    "ASCT2":  {"s1": 0.699060, "s2": 0.871854},
    "CCR5":   {"s1": 0.976848, "s2": 0.921632},
    "CGRPR":  {"s1": 0.879535, "s2": 0.939928},
    "FZD7":   {"s1": 0.925391, "s2": 0.935591},
    "LAT1":   {"s1": 0.944684, "s2": 0.905509},
    "MCT1":   {"s1": 0.972060, "s2": 0.804134},
    "MurJ":   {"s1": 0.962784, "s2": 0.737712},
    "PTH1R":  {"s1": 0.851181, "s2": 0.938626},
    "PfMATE": {"s1": 0.731453, "s2": 0.992106},
    "SERT":   {"s1": 0.910615, "s2": 0.985340},
    "STP10":  {"s1": 0.913978, "s2": 0.916406},
    "ZnT8":   {"s1": 0.887613, "s2": 0.884766}
}


def parse_ablation_details(experiment_name, nseq):
    """
    Robust parser that extracts seeds and keeps ablation levels numeric.
    Guaranteed to return a 3-tuple (exp_type, val, seed) on all code paths.
    """
    exp_str = str(experiment_name).strip()
    
    # 1. CATCH THE BASELINE / CONTROL FOLDER
    if exp_str.lower() in ["base_case", "control", "baseline", "nan", "no experiment type"]:
        return "Baseline", 0, "Seed0"

    name_clean = exp_str.replace("_", " ").title()
    
    # Extract Seed if present (e.g., "Seed 0" or "Seed0" or "Seed1")
    seed_match = re.search(r'Seed\s*(\d+)', name_clean, re.IGNORECASE)
    seed = f"Seed{seed_match.group(1)}" if seed_match else "Seed0"
    
    # Remove seed noise to parse the ablation value cleanly
    name_no_seed = re.sub(r'Seed\s*\d+', '', name_clean, flags=re.IGNORECASE).strip()

    # Find the FIRST number in the string (this is your clean ablation level)
    match = re.search(r'(\d+)', name_no_seed)
    
    if match:
        val = int(match.group(1)) # Keep it strictly as an integer for math/sorting
        
        # Standardize naming mapping
        if "Query" in name_clean:
            exp_type = "Query Mask"
        elif "Row" in name_clean:
            exp_type = "Row Mask"
        elif "Col" in name_clean:
            exp_type = "Col Mask"
        elif "Depth" in name_clean:
            exp_type = "Depth"
        else:
            exp_type = re.sub(r'\d+.*', '', name_no_seed).strip()
            if not exp_type:
                exp_type = "Experiment"
                
        return exp_type, val, seed
        
    # 3. FALLBACK TO NSEQ IF RUNNING DEPTH EXPERIMENTS
    if "Depth" in name_clean or nseq != 5120:
        return "MSA Depth Reduction", int(nseq), seed
        
    return "NaN", 0, seed


def process_evaluations(data_files: list[str], metric: str = "max", top_k: int = 5) -> pd.DataFrame:
    dfs = []
    has_if_labels = False
    has_ia_labels = False

    for f in data_files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue

            # Force parse_ablation_details to look at the FOLDER name, not the internal CSV column
            subfolder_name = Path(f).parent.name
            df["folder_experiment"] = subfolder_name

            if "protein" not in df.columns or df["protein"].dropna().empty:
                df["protein"] = subfolder_name

            # Standardize headers per file before concatenating
            if "tm_IF" in df.columns:
                df = df.rename(columns={"tm_IF": "state_1", "tm_OF": "state_2"})
                has_if_labels = True
            elif "mean_tm_IF" in df.columns:
                df = df.rename(columns={"mean_tm_IF": "state_1", "mean_tm_OF": "state_2"})
                has_if_labels = True
            elif "tm_I" in df.columns:
                df = df.rename(columns={"tm_I": "state_1", "tm_A": "state_2"})
                has_ia_labels = True
            else:
                continue # Skip unknown file formats

            dfs.append(df)
        except Exception:
            continue
            
    if not dfs:
        raise ValueError("No valid or non-empty CSV files found.")
        
    df = pd.concat(dfs, ignore_index=True)
    
    if has_if_labels and has_ia_labels:
        s1_label, s2_label = "State 1 (IF/I)", "State 2 (OF/A)"
    elif has_if_labels:
        s1_label, s2_label = "Inward (IF)", "Outward (OF)"
    else:
        s1_label, s2_label = "Inactive (I)", "Active (A)"

    s1_col, s2_col = "state_1", "state_2"

    # Use 'folder_experiment' instead of 'experiment' to ensure Seed is never ignored
    parsed = df.apply(lambda row: parse_ablation_details(row["folder_experiment"], row["nseq"]), axis=1)
    df["exp_type"] = [p[0] for p in parsed]
    df["ablation_val"] = [p[1] for p in parsed]
    df["seed"] = [p[2] for p in parsed]

    # --- DEBUGGING PRINTS ---
    print("\n" + "-"*50)
    print("[DIAGNOSTIC] Successfully parsed unique combinations:")
    print(df[["folder_experiment", "exp_type", "ablation_val", "seed"]].drop_duplicates().to_string(index=False))
    print("-"*50 + "\n")

    # Group and aggregate stats, including seed in grouping keys
    group_cols = ["exp_type", "ablation_val", "seed", "protein"]
    
    def mean_top_k(x):
        return x.nlargest(top_k).mean()

    if metric == "max":
        stats = df.groupby(group_cols).agg(
            s1_min=(s1_col, "min"), s1_max=(s1_col, "max"), s1_mean=(s1_col, "mean"),
            s2_min=(s2_col, "min"), s2_max=(s2_col, "max"), s2_mean=(s2_col, "mean")
        ).reset_index()
        s1_target_col, s2_target_col = "s1_max", "s2_max"
    elif metric == "mean":
        stats = df.groupby(group_cols).agg(
            s1_min=(s1_col, "min"), s1_max=(s1_col, "max"), s1_mean=(s1_col, "mean"),
            s2_min=(s2_col, "min"), s2_max=(s2_col, "max"), s2_mean=(s2_col, "mean")
        ).reset_index()
        s1_target_col, s2_target_col = "s1_mean", "s2_mean"
    elif metric == "top_k":
        stats = df.groupby(group_cols).agg(
            s1_min=(s1_col, "min"), s1_max=(s1_col, "max"), s1_mean=(s1_col, "mean"), s1_top5=(s1_col, mean_top_k),
            s2_min=(s2_col, "min"), s2_max=(s2_col, "max"), s2_mean=(s2_col, "mean"), s2_top5=(s2_col, mean_top_k)
        ).reset_index()
        s1_target_col, s2_target_col = "s1_top5", "s2_top5"

    stats = stats.dropna(subset=["s1_mean", "s2_mean"])

    def get_delta(row, state_key, target_col):
        protein_key = str(row["protein"]).strip()
        if protein_key in BASELINE_LOOKUP:
            if metric == "max":
                base_val = BASELINE_LOOKUP[protein_key][state_key]
            elif metric == "mean":
                base_val = MEAN_LOOKUP[protein_key][state_key]
            elif metric == "top_k":
                base_val = TOP_5_BASELINE[protein_key][state_key]
            else:
                base_val = 0.1
            return row[target_col] - base_val
        return 0.0

    stats["s1_delta"] = stats.apply(lambda r: get_delta(r, "s1", s1_target_col), axis=1)
    stats["s2_delta"] = stats.apply(lambda r: get_delta(r, "s2", s2_target_col), axis=1)

    if metric == "top_5":
        stats = stats.drop(columns=["s1_top5", "s2_top5"])
    return stats, (s1_label, s2_label)


def generate_markdown_reports(stats, labels, output_path=None):
    s1_lbl, s2_lbl = labels
    report_str = []
    
    report_str.append("# Automated Ablation Experiment Evaluation Report\n")

    unified_report = stats.copy()
    
    # Sort strictly using the lowercase column names
    unified_report = unified_report.sort_values(by=["protein", "exp_type", "seed", "ablation_val"])

    print("DEBUG - actual columns:", list(unified_report.columns))

# 1. First, reorder the raw columns using the actual database keys (so they never mismatch)
    raw_column_order = [
        "protein", "exp_type", "ablation_val", "seed",
        "s1_min", "s1_max", "s1_mean", "s1_top5", "s1_delta",
        "s2_min", "s2_max", "s2_mean", "s2_top5", "s2_delta"
    ]
    
    # Slice the dataframe safely using original keys
    existing_columns = [col for col in raw_column_order if col in unified_report.columns]
    unified_report = unified_report[existing_columns]

    # 2. Map the keys to their final pretty names
    rename_map = {
        "protein": "Protein",
        "exp_type": "Experiment Type",
        "ablation_val": "Ablation Level",
        "seed": "Seed",
        "s1_min": f"Min {s1_lbl}",
        "s1_max": f"Max {s1_lbl}",
        "s1_mean": f"Mean {s1_lbl}",
        "s1_top5": f"Top 5 {s1_lbl}",
        "s1_delta": f"Δ Base {s1_lbl}",
        "s2_min": f"Min {s2_lbl}",
        "s2_max": f"Max {s2_lbl}",
        "s2_mean": f"Mean {s2_lbl}",
        "s2_top5": f"Top 5 {s2_lbl}",
        "s2_delta": f"Δ Base {s2_lbl}"
    }
    
    # 3. Rename them safely
    unified_report = unified_report.rename(columns=rename_map)
    
    report_str.append(unified_report.to_markdown(index=False, floatfmt=".3f"))
    final_output = "\n".join(report_str)
    
    if output_path:
        Path(output_path).write_text(final_output)
        print(f"[Success] Unified report saved to: {output_path}")
    else:
        print(final_output)


def main():
    default_filename = f"./ablation_report_{datetime.now().strftime('%Y-%m-%d')}.md"

    class DynamicHelpAction(argparse._HelpAction):
        def __call__(self, parser, namespace, values, option_string=None):
            data_dir = "./"
            for i, arg in enumerate(sys.argv):
                if arg in ["-i", "--data_dir"]:
                    if i + 1 < len(sys.argv):
                        data_dir = sys.argv[i + 1]
                elif arg.startswith("--data_dir="):
                    data_dir = arg.split("=", 1)[1]
            
            subfolders = []
            path = Path(data_dir)
            if path.exists() and path.is_dir():
                for root, dirs, _ in os.walk(path, followlinks=True):
                    rel = os.path.relpath(root, path)
                    if rel != ".":
                        subfolders.append(f"    - {rel}")
            
            if subfolders:
                folder_list_str = f"\nAvailable subfolders found in '{data_dir}':\n" + "\n".join(subfolders)
                # Direct access with 'or' fallback, no getattr needed
                parser.epilog = (parser.epilog or '') + folder_list_str
            else:
                # Direct access with 'or' fallback, no getattr needed
                parser.epilog = (parser.epilog or '') + f"\n\nNo subfolders discovered in '{data_dir}'."
            
            super().__call__(parser, namespace, values, option_string)

    parser = argparse.ArgumentParser(
        description="Process ablation experiment evaluation logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
    )
    
    parser.add_argument('-h', '--help', action=DynamicHelpAction, help='show help')
    parser.add_argument("-i", "--data_dir", type=str, default="./")
    parser.add_argument("-o", "--output_md", type=str, default=default_filename)
    parser.add_argument("--experiment_include", type=str, nargs="+")
    parser.add_argument("--experiment_exclude", type=str, nargs="+")
    parser.add_argument("--compare", type=str, choices=["max", "mean", "top_k"], default="max")
    parser.add_argument("-k", "--top_k", type=int, default=5)

    args = parser.parse_args()
    
    try:
        input_path = Path(args.data_dir)
        csv_files = []
        
        for root, dirs, files in os.walk(input_path, followlinks=True):
            folder_name = os.path.relpath(root, input_path)
            if folder_name != ".":
                if args.experiment_exclude and any(x.lower() in folder_name.lower() for x in args.experiment_exclude):
                    continue
                if args.experiment_include:
                    match_found = False
                    for x in args.experiment_include:
                        x_low = x.lower()
                        f_low = folder_name.lower()
                        if any(c.isdigit() for c in x_low):
                            top_level_folder = f_low.replace('\\', '/').split('/')[0]
                            if x_low == top_level_folder:
                                match_found = True
                                break
                        else:
                            if x_low in f_low:
                                match_found = True
                                break
                    if not match_found:
                        continue

            for file in files:
                if file.endswith(".csv"):
                    csv_files.append(os.path.join(root, file))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {args.data_dir}")

        stats_df, labels = process_evaluations(csv_files, metric=args.compare, top_k=args.top_k)
        generate_markdown_reports(stats_df, labels, args.output_md)
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"[CRITICAL ERROR] Failed to complete ablation evaluation execution.")
        print(f"Error Type:    {type(e).__name__}")
        print(f"Error Summary: {e}")
        print("="*60)
        print("Detailed Execution Traceback:")
        traceback.print_exc()
        print("="*60 + "\n")

if __name__ == "__main__":
    main()