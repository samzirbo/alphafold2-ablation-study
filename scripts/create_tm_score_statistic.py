import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np
from datetime import datetime
import traceback


def parse_ablation_details(experiment_name, nseq):
    """
    Improved parser that links the baseline folder directly to your 
    ablation experiment types to ensure groups and deltas compute correctly.
    """
    # Convert to string and handle standard edge cases
    exp_str = str(experiment_name).strip()
    
    # 1. CATCH THE BASELINE / CONTROL FOLDER
    # Adjust these strings if your baseline directory is named differently

    if exp_str.lower() in ["base_case", "control", "baseline", "nan", "depth_5120", "no experiment type"]:
        # We default it to the primary experiment type we are analyzing
        return "Baseline", 0

    # 2. PARSE THE ABLATION LEVEL FROM FOLDER NAME (e.g., 'query_masking_15' -> 15)
    name_clean = exp_str.replace("_", " ").title()
    match = re.search(r'(\d+)\s*%', name_clean) or re.search(r'(\d+)$', name_clean)
    
    if match:
        val = int(match.group(1))
        # Strip the trailing digits to get the clean experiment type
        exp_type = re.sub(r'\d+\s*%?$', '', name_clean).strip()
        
        # Standardize naming mapping
        if "Query" in exp_type:
            exp_type = "Query Mask"
        return exp_type, val
        
    # 3. FALLBACK TO NSEQ IF RUNNING DEPTH EXPERIMENTS
    if "Depth" in name_clean or nseq != 5120:
        return "MSA Depth Reduction", int(nseq)
        
    return "NaN", 0


def process_evaluations(data_files: list[str]) -> pd.DataFrame:
    dfs = []
    for f in data_files:

        df = pd.read_csv(f)

        subfolder_name = Path(f).parent.name
        # Example: If 'expermint is missing (older plotter didnt include it), fall back to 
        # the subfolder nam e
        if "experiment" not in df.columns or df["experiment"].dropna().empty:
                df["experiment"] = subfolder_name
        dfs.append(df)

        
    df = pd.concat(dfs, ignore_index=True)
    
    # Drop rows that have NaN values in vital TM score paths
    if "tm_IF" in df.columns:
        s1_col, s2_col = "tm_IF", "tm_OF"
        s1_label, s2_label = "Inward (IF)", "Outward (OF)"
    elif "tm_I" in df.columns:
        s1_col, s2_col = "tm_I", "tm_A"
        s1_label, s2_label = "Inactive (I)", "Active (A)"
    else:
        raise ValueError("CSV structure not recognized. Missing expected TM columns.")

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

    # Compute Delta Change relative to baseline (Ablation Level == 0)
    base_cases = stats[stats["ablation_val"] == 0].set_index(["exp_type", "protein"])

    def get_delta(row, state_mean_col):
        key = (row["exp_type"], row["protein"])
        if key in base_cases.index:
            base_row = base_cases.loc[[key]]
            base_mean = base_row[state_mean_col].values[0]
            return row[state_mean_col] - base_mean
        return 0.0

    stats["s1_delta"] = stats.apply(lambda r: get_delta(r, "s1_mean"), axis=1)
    stats["s2_delta"] = stats.apply(lambda r: get_delta(r, "s2_mean"), axis=1)
    
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
    default_filename = f"ablation_report_{datetime.now().strftime('%Y-%m-%d')}.md"

    usage_examples = f"""
Examples of usage:
  # 1. Omitting output saves automatically to current directory (e.g., {default_filename})
  python %(prog)s -i data/*.csv

  # 2. Specifying a custom output path
  python %(prog)s -i data/*.csv -o custom_report.md


Output Report Columns Dictionary:
    Experiment Type       The categorized track evaluated (e.g., Query Mask, MSA Depth Reduction).
    Ablation Level        Integer degree of reduction applied (e.g., 0 for Baseline/Control, 15 for 15 percent)
    Protein               The targeted system identifier extracted from the trial data.
    Min [State]           The minimum observed TM score for that specific state configuration group.
    Max [State]           The peak observed TM score achievement.
    Mean [State]          Arithmetic average of structural scores across matching runs.
    Δ Base [State]        Mathematical absolute difference comparing current Mean score to the matching 
                        Baseline run (Ablation Level == 0). Negative implies degradation.
    """

    parser = argparse.ArgumentParser(
        description="Process ablation experiment evaluation logs.",
        epilog=usage_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--data_dir", 
        type=str, 
        required=True,
        help="Path to the directory containing evaluation CSV files."
    )
    
    parser.add_argument(
        "-o", "--output_md", 
        type=str, 
        default=default_filename,
        help=f"Path to output markdown file. Defaults to '{default_filename}' if omitted."
    )
    
    args = parser.parse_args()
    
    try:
        
        input_path = Path(args.data_dir)
        csv_files = [str(p) for p in input_path.rglob("*.csv")]

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