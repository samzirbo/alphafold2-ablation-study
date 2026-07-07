import argparse
from pathlib import Path
import io
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_numeric_suffix(condition_string):
    """
    Extracts numbers safely from a condition string to ensure numerical sorting.
    e.g., 'Lvl 16' -> 16, 'Baseline' -> 0
    """
    match = re.search(r'\d+', str(condition_string))
    return int(match.group()) if match else 0


def generate_heatmap_from_md(md_file_path: str, output_image_path: str = "ablation_delta_heatmap.png", show_annotations: bool = False):
    path = Path(md_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find the specified Markdown report at: {md_file_path}")
        
    print(f"[Info] Scanning file line-by-line for Markdown tables: {md_file_path}")
    
    table_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                # Skip markdown separator rows
                if any(x in stripped for x in ["---", "-:", ":-"]):
                    continue
                table_lines.append(stripped)
                
    if not table_lines:
        raise ValueError("Could not find any valid Markdown table lines starting and ending with '|' in the file.")
    
    # Load into Pandas DataFrame
    clean_table_str = "\n".join(table_lines)
    df = pd.read_csv(io.StringIO(clean_table_str), sep="|", engine="python")
    
    # Clean up markdown artifact columns and structural spacing
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(how="all", axis=0)
    
    # Ensure all string columns are completely stripped of hidden whitespace
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()
            
    # Explicitly enforce numeric values
    df["Ablation Level"] = pd.to_numeric(df["Ablation Level"], errors='coerce')
    df["Δ Base State 1 (IF/I)"] = pd.to_numeric(df["Δ Base State 1 (IF/I)"], errors='coerce')
    df["Δ Base State 2 (OF/A)"] = pd.to_numeric(df["Δ Base State 2 (OF/A)"], errors='coerce')

    df.rename(
        columns={
            "Δ Base State 1 (IF/I)": "Delta_State1",
            "Δ Base State 2 (OF/A)": "Delta_State2",
        },
        inplace=True,
    )

    # Isolate base conditions vs experimental changes
    baseline_mask = df["Experiment Type"].str.lower() == "baseline"
    baseline_df = df[baseline_mask].copy()
    experiment_df = df[~baseline_mask].copy()
    
    unique_experiments = sorted(experiment_df["Experiment Type"].unique())
    num_experiments = len(unique_experiments)

    if num_experiments == 0:
        raise ValueError("No distinct experiment types found outside of Baseline rows.")

    # Setup visualization grid
    fig, axes = plt.subplots(num_experiments, 2, figsize=(14, 4 * num_experiments), squeeze=False)
    cmap = "coolwarm"  

    # Establish global symmetric bounds centered at 0.0 for accurate color comparisons
    max_val = max(df["Delta_State1"].abs().max(), df["Delta_State2"].abs().max())
    if pd.isna(max_val) or max_val == 0:
        max_val = 1.0  # Fallback boundary
    vmin, vmax = -max_val, max_val

    for idx, exp_type in enumerate(unique_experiments):
        ax_s1 = axes[idx, 0]
        ax_s2 = axes[idx, 1]
        
        # Extract experiment rows
        exp_data = experiment_df[experiment_df["Experiment Type"] == exp_type].copy()
        exp_data["Y_Label"] = "Lvl " + exp_data["Ablation Level"].astype(int, errors='ignore').astype(str)
        
        # Mirror the baseline records specifically scaled to align with this experiment group
        current_baseline = baseline_df.copy()
        current_baseline["Experiment Type"] = exp_type
        current_baseline["Y_Label"] = "Baseline"
        
        # Combine structural records together
        combined_block = pd.concat([exp_data, current_baseline], ignore_index=True)
        
        # Generate clean 2D Matrices via pivot maps
        pivot_s1 = combined_block.pivot_index_or_values(index="Y_Label", columns="Protein", values="Delta_State1") if hasattr(combined_block, 'pivot_index_or_values') else combined_block.pivot(index="Y_Label", columns="Protein", values="Delta_State1")
        pivot_s2 = combined_block.pivot(index="Y_Label", columns="Protein", values="Delta_State2")
        
        # Handle hierarchical sort logic securely (Levels 1 to N sequentially, Baseline anchored at the bottom)
        custom_order = sorted(
            combined_block["Y_Label"].unique(),
            key=lambda x: (1 if "Baseline" in str(x) else 0, get_numeric_suffix(x)),
        )
        pivot_s1 = pivot_s1.reindex(custom_order)
        pivot_s2 = pivot_s2.reindex(custom_order)


        # Plot State 1 Matrix
        sns.heatmap(
            pivot_s1, annot=show_annotations, fmt=".3f", cmap=cmap, vmin=vmin, vmax=vmax,
            ax=ax_s1, cbar=False, linewidths=0.5, annot_kws={"size": 9}
        )
        ax_s1.set_ylabel(exp_type, fontsize=11, weight="bold")
        
        # Plot State 2 Matrix (only display layout legend bar at the bottom right corner)
        show_cbar = (idx == num_experiments - 1)
        cbar_kwargs = {"label": "Change Relative to Baseline (Δ)", "pad": 0.03} if show_cbar else {}
        
        sns.heatmap(
            pivot_s2, annot=show_annotations, fmt=".3f", cmap=cmap, vmin=vmin, vmax=vmax,
            ax=ax_s2, cbar=show_cbar, cbar_kws=cbar_kwargs, linewidths=0.5, annot_kws={"size": 9}
        )
        ax_s2.set_ylabel("")

        # Formatting top-level titles
        if idx == 0:
            ax_s1.set_title("Δ TM-Score: State 1 (IF/I)", fontsize=12, weight="bold", pad=12)
            ax_s2.set_title("Δ TM-Score: State 2 (OF/A)", fontsize=12, weight="bold", pad=12)

        # Handle X-axis label positioning explicitly
        if idx == num_experiments - 1:
            ax_s1.set_xlabel("Protein Target", fontsize=11, labelpad=10)
            ax_s2.set_xlabel("Protein Target", fontsize=11, labelpad=10)
            plt.setp(ax_s1.get_xticklabels(), rotation=45, ha="right")
            plt.setp(ax_s2.get_xticklabels(), rotation=45, ha="right")
        else:
            ax_s1.set_xlabel("")
            ax_s2.set_xlabel("")
            ax_s1.get_xaxis().set_visible(False)
            ax_s2.get_xaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    print(f"[Success] Grouped Experiment Heatmap saved to: {output_image_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate structural delta heatmaps grouped by experiment category.")
    parser.add_argument("-i", "--input_md", type=str, required=True, help="Path to your .md file.")
    parser.add_argument("-o", "--output_png", type=str, default="ablation_delta_heatmap.png", help="Output destination image path.")
    parser.add_argument("-sa", "--show_annot", action="store_false", help="Show numerical text annotations inside the heatmap cells.")
    args = parser.parse_args()

    try:
        generate_heatmap_from_md(args.input_md, args.output_png, show_annotations = not args.show_annot)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to plot grouped heatmap: {e}")


if __name__ == "__main__":
    main()