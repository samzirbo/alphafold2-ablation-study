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
# Potentially mean values or any other metric can also be applied here! 
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



def parse_ablation_details(experiment_name, nseq):
    """
    Improved parser that links the baseline folder directly to your 
    ablation experiment types to ensure groups and deltas compute correctly.
    """
    # Convert to string and handle standard edge cases
    exp_str = str(experiment_name).strip()
    
    # 1. CATCH THE BASELINE / CONTROL FOLDER
    # Adjust these strings if your baseline directory is named differently

    if exp_str.lower() in ["base_case", "control", "baseline", "nan", "no experiment type"]:
        # We default it to the primary experiment type we are analyzing
        return "Baseline", 0

    # 2. PARSE THE ABLATION LEVEL FROM FOLDER NAME (e.g., 'query_masking_15' -> 15)
    name_clean = exp_str.replace("_", " ").title()
    # Find the FIRST number in the string (this correctly grabs 5 from 'Query Mask 5 Seed 0')
    match = re.search(r'(\d+)', name_clean)
    
    if match:
        val = int(match.group(1))
        
        # Standardize naming mapping robustly based on keywords
        if "Query" in name_clean:
            exp_type = "Query Mask"
        elif "Row" in name_clean:
            exp_type = "Row Mask"
        elif "Col" in name_clean:
            exp_type = "Col Mask"
        elif "Depth" in name_clean:
            exp_type = "Depth"
        else:
            # Fallback for unknown patterns: strip from the first digit onwards
            exp_type = re.sub(r'\d+.*', '', name_clean).strip()
            if not exp_type:
                exp_type = "Experiment"
                
        return exp_type, val
        
    # 3. FALLBACK TO NSEQ IF RUNNING DEPTH EXPERIMENTS
    if "Depth" in name_clean or nseq != 5120:
        return "MSA Depth Reduction", int(nseq)
        
    return "NaN", 0


def process_evaluations(data_files: list[str]) -> pd.DataFrame:
    dfs = []
    has_if_labels = False
    has_ia_labels = False

    for f in data_files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue

            subfolder_name = Path(f).parent.name
            if "experiment" not in df.columns or df["experiment"].dropna().empty:
                df["experiment"] = subfolder_name

            if "protein" not in df.columns or df["protein"].dropna().empty:
                print(f"[WARNING] Missing 'protein' header in file: '{f}'")
                print(f"          -> Fallback applied: Assigning protein tracking label as '{subfolder_name}'\n")
                df["protein"] = subfolder_name

            # Fix: Standardize headers per file BEFORE pd.concat
            if "tm_IF" in df.columns:
                df = df.rename(columns={"tm_IF": "state_1", "tm_OF": "state_2"})
                has_if_labels = True
            elif "tm_I" in df.columns:
                df = df.rename(columns={"tm_I": "state_1", "tm_A": "state_2"})
                has_ia_labels = True
            else:
                available_cols = list(df.columns)
                print(f"[WARNING] Dropping file: '{f}'")
                print(f"          Reason: Could not find expected TM structural headers.")
                print(f"          Found columns: {available_cols}")
                print(f"          Expected: ['tm_IF', 'tm_OF'] OR ['tm_I', 'tm_A']\n")
                continue # Skip unknown file formats safely

            dfs.append(df)
        except Exception:
            continue
            
    if not dfs:
        raise ValueError("No valid or non-empty CSV files found.")
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Replace the old if/elif block with this clean dynamic label selector:
    if has_if_labels and has_ia_labels:
        s1_label, s2_label = "State 1 (IF/I)", "State 2 (OF/A)"
    elif has_if_labels:
        s1_label, s2_label = "Inward (IF)", "Outward (OF)"
    else:
        s1_label, s2_label = "Inactive (I)", "Active (A)"

    s1_col, s2_col = "state_1", "state_2"

    # Apply the corrected parsing function
    parsed = df.apply(lambda row: parse_ablation_details(row["experiment"], row["nseq"]), axis=1)
    df["exp_type"] = [p[0] for p in parsed]
    df["ablation_val"] = [p[1] for p in parsed]

    # Group and aggregate stats, omitting NaN predictions automatically
    group_cols = ["exp_type", "ablation_val", "protein"]
    stats = df.groupby(group_cols).agg(
        s1_min=(s1_col, "min"), s1_max=(s1_col, "max"), s1_mean=(s1_col, "mean"),
        s2_min=(s2_col, "min"), s2_max=(s2_col, "max"), s2_mean=(s2_col, "mean")
    ).reset_index()

    # Drop any protein aggregates that ended up completely empty (e.g. CCR5, CGRPR)
    stats = stats.dropna(subset=["s1_mean", "s2_mean"])

# 1. Isolate the baselines using ONLY the protein name as the unique key
    # (Assuming every protein has exactly one baseline entry where ablation_val == 0)
    baselines_df = stats[stats["ablation_val"] == 0].set_index("protein")

    def get_delta(row, state_key):
        protein_key = str(row["protein"]).strip()
        max_col = "s1_max" if state_key == "s1" else "s2_max"
        
        # Pull baseline value directly from hardcoded object dictionary
        if protein_key in BASELINE_LOOKUP:
            base_max = BASELINE_LOOKUP[protein_key][state_key]
            return row[max_col] - base_max
            
        return 0.0

    stats["s1_delta"] = stats.apply(lambda r: get_delta(r, "s1"), axis=1)
    stats["s2_delta"] = stats.apply(lambda r: get_delta(r, "s2"), axis=1)
    
    return stats, (s1_label, s2_label)

def generate_markdown_reports(stats, labels, output_path=None):
    s1_lbl, s2_lbl = labels
    report_str = []
    
    report_str.append("# Automated Ablation Experiment Evaluation Report\n")


# Create a unified copy and sort strictly by Protein first
    unified_report = stats.sort_values(by=["protein", "exp_type", "ablation_val"]).copy()
    
    # Arrange and rename columns into a single, cohesive master table
    unified_report.columns = [
        "Experiment Type", "Ablation Level", "Protein",
        f"Min {s1_lbl}", f"Max {s1_lbl}", f"Mean {s1_lbl}",
        f"Min {s2_lbl}", f"Max {s2_lbl}", f"Mean {s2_lbl}",
        f"Δ Base {s1_lbl}", f"Δ Base {s2_lbl}"
    ]
    
    # Reorder columns to put 'Protein' upfront as the primary anchor
    column_order = [
        "Protein", "Experiment Type", "Ablation Level",
        f"Min {s1_lbl}", f"Max {s1_lbl}", f"Mean {s1_lbl}", f"Δ Base {s1_lbl}",
        f"Min {s2_lbl}", f"Max {s2_lbl}", f"Mean {s2_lbl}", f"Δ Base {s2_lbl}"
    ]
    unified_report = unified_report[column_order]
    
    # Convert to markdown layout
    report_str.append(unified_report.to_markdown(index=False, floatfmt=".3f"))
    
    final_output = "\n".join(report_str)
    
    if output_path:
        Path(output_path).write_text(final_output)
        print(f"[Success] Unified report saved to: {output_path}")
    else:
        print(final_output)

  


def main():
    # Generate a dynamic default filename with a timestamp


    default_filename = f"/content/drive/MyDrive/AlphaFold2 Ablation Study/04_Results/plots/TM_Score/ablation_report_{datetime.now().strftime('%Y-%m-%d')}.md"

    class DynamicHelpAction(argparse._HelpAction):
        def __call__(self, parser, namespace, values, option_string=None):
            # Fallback default path
            data_dir = "/content/drive/MyDrive/AlphaFold2 Ablation Study/04_Results/plots/TM_Score/"
            
            # Manually extract the data directory from raw sys.argv if present
            # Checks for both '-i value' and '--data_dir value' or '--data_dir=value'
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
                parser.epilog = getattr(parser, 'epilog', '') + folder_list_str
            else:
                parser.epilog = getattr(parser, 'epilog', '') + f"\n\nNo subfolders discovered in '{data_dir}' (or directory doesn't exist yet)."
            
            super().__call__(parser, namespace, values, option_string)


    usage_examples = f"""
Examples of usage:
  # 1. Omitting output saves automatically to current directory (e.g., {default_filename})
  python %(prog)s -i data/

  # 2. Specifying a custom output path
  python %(prog)s -i data/ -o custom_report.md

  # 3. Match multiple patterns (e.g., must contain 'query' OR 'row')
  python %(prog)s -i data/ --experiment_include query row


Output Report Columns Dictionary:
    Experiment Type       The categorized track evaluated (e.g., Query Mask, MSA Depth Reduction).
    Ablation Level        Integer degree of reduction applied (e.g., 0 for Baseline/Control, 15 for 15 percent)
    Protein               The targeted system identifier extracted from the trial data.
    Min [State]           The minimum observed TM score for that specific state configuration group.
    Max [State]           The peak observed TM score achievement.
    Mean [State]          Arithmetic average of structural scores across matching runs.
    Δ Base [State]        Mathematical absolute difference comparing current Max score to the matching 
                        Baseline run (Ablation Level == 0). Negative implies degradation.
    """

    parser = argparse.ArgumentParser(
        description="Process ablation experiment evaluation logs.",
        epilog=usage_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
    )
    

    parser.add_argument(
        '-h', '--help', 
        action=DynamicHelpAction, 
        help='show this help message and exit'
    )
    parser.add_argument(
        "-i", "--data_dir", 
        type=str, 
        default="/content/drive/MyDrive/AlphaFold2 Ablation Study/04_Results/plots/TM_Score/",
        help="Path to the directory containing evaluation CSV files."
    )
    
    parser.add_argument(
        "-o", "--output_md", 
        type=str, 
        default=default_filename,
        help=f"Path to output markdown file. Defaults to '{default_filename}' if omitted."
    )

    parser.add_argument(
        "--experiment_include",
        type=str,
        nargs="+",
        help="One or more substrings. Only process subfolders that contain at least one of these strings."
    )
    parser.add_argument(
        "--experiment_exclude",
        type=str,
        nargs="+",
        help= "One or more substrings. Completely skip subfolders containing any of these strings."
    )

    
    args = parser.parse_args()
    
    try:
        
        input_path = Path(args.data_dir)

        # ─── FIXED GOOGLE DRIVE FILE SEARCH START ───

        csv_files = []
        
        # os.walk with followlinks=True forces Google Drive to open and read subfolders
        for root, dirs, files in os.walk(input_path, followlinks=True):

            folder_name = os.path.relpath(root, input_path)

            # Skip base dir
            if folder_name != ".":
                #Exclusion: Skip any exclude keywors
                if args.experiment_exclude and any(x.lower() in folder_name.lower() for x in args.experiment_exclude):
                    continue

                # Skip if no include match 
                if args.experiment_include:
                    match_found = False
                    for x in args.experiment_include:
                        x_low = x.lower()
                        f_low = folder_name.lower()
                        
                        # Specific fix: If include string contains a digit, require exact match with top-level folder
                        # to prevent 'query_mask_5' from matching 'query_mask_50' or 'query_mask_5_Seed0'
                        if any(c.isdigit() for c in x_low):
                            top_level_folder = f_low.replace('\\', '/').split('/')[0]
                            if x_low == top_level_folder:
                                match_found = True
                                break
                        else:
                            # Generic pattern (e.g., 'query' or 'row') -> allow substring match
                            if x_low in f_low:
                                match_found = True
                                break
                                
                    if not match_found:
                        continue

        
                

            for file in files:
                if file.endswith(".csv"):
                    full_path = os.path.join(root, file)
                    csv_files.append(full_path)
        # ─── FIXED GOOGLE DRIVE FILE SEARCH END ───

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {args.data_dir}")

        #print(csv_files)

        stats_df, labels = process_evaluations(csv_files)

        
        generate_markdown_reports(stats_df, labels, args.output_md)
    except Exception as e:
        # Verbose Error Block
        print("\n" + "="*60)
        print(f"[CRITICAL ERROR] Failed to complete ablation evaluation execution.")
        print(f"Error Type:    {type(e).__name__}")
        print(f"Error Summary: {e}")
        print("="*60)
        print("Detailed Execution Traceback:")
        traceback.print_exc()  # Prints the full stack trace directly to stderr
        print("="*60 + "\n")

if __name__ == "__main__":
    main()