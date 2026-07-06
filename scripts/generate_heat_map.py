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
    e.g., 'Depth (Lvl 16)' -> 16, 'Depth (Lvl 1024)' -> 1024
    """
    match = re.search(r'\d+', condition_string)
    return int(match.group()) if match else 0


def generate_heatmap_from_md(md_file_path: str, output_image_path: str = "ablation_delta_heatmap.png"):
    path = Path(md_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find the specified Markdown report at: {md_file_path}")
        
    print(f"[Info] Scanning file line-by-line for Markdown tables: {md_file_path}")
    
    # Isolate table lines safely to bypass metadata or description rows
    table_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                if any(x in stripped for x in ["---", "-:", ":-"]):
                    continue
                table_lines.append(stripped)
                
    if not table_lines:
        raise ValueError("Could not find any valid Markdown table lines starting and ending with '|' in the file.")
    
    # Load into Pandas DataFrame
    clean_table_str = "\n".join(table_lines)
    df = pd.read_csv(io.StringIO(clean_table_str), sep="|", engine="python")
    
    # Clean up markdown artifact columns and padding
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(how="all", axis=0)
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    
    # Direct explicit numeric enforcement
    df["Ablation Level"] = pd.to_numeric(df["Ablation Level"])
    df["Δ Base State 1 (IF/I)"] = pd.to_numeric(df["Δ Base State 1 (IF/I)"])
    df["Δ Base State 2 (OF/A)"] = pd.to_numeric(df["Δ Base State 2 (OF/A)"])

    df.rename(
        columns={
            "Δ Base State 1 (IF/I)": "Delta_State1",
            "Δ Base State 2 (OF/A)": "Delta_State2",
        },
        inplace=True,
    )

    # Separate baseline rows so we can append them cleanly to each experiment block
    baseline_df = df[df["Experiment Type"].str.lower() == "baseline"].copy()
    experiment_df = df[df["Experiment Type"].str.lower() != "baseline"].copy()
    
    unique_experiments = sorted(experiment_df["Experiment Type"].unique())
    num_experiments = len(unique_experiments)

    if num_experiments == 0:
        raise ValueError("No distinct experiment types found outside of Baseline rows.")

    # Dynamically scale figure height based on the number of experiment types
    fig, axes = plt.subplots(num_experiments, 2, figsize=(16, 4 * num_experiments), sharex=True, squeeze=False)
    cmap = "coolwarm"  

    # Balance diverging colorbar limits perfectly around 0.0
    max_val = max(abs(df["Delta_State1"].max()), abs(df["Delta_State2"].max()))
    vmin, vmax = -max_val, max_val

    # Loop through each experiment type to build grouped stacked subplots
    for idx, exp_type in enumerate(unique_experiments):
        ax_s1 = axes[idx, 0]
        ax_s2 = axes[idx, 1]
        
        # Pull data for this specific experiment and combine it with baseline rows for comparison
        exp_data = experiment_df[experiment_df["Experiment Type"] == exp_type].copy()
        
        # Inject current experiment's label into baseline copies so they map onto the pivot tables correctly
        current_baseline = baseline_df.copy()
        current_baseline["Experiment Type"] = exp_type
        
        combined_block = pd.concat([exp_data, current_baseline], ignore_index=True)
        combined_block["Y_Label"] = "Lvl " + combined_block["Ablation Level"].astype(str)
        
        # Force baseline label to look clean at the bottom
        combined_block.loc[combined_block["Experiment Type"].str.lower() == "baseline", "Y_Label"] = "Baseline"
        
        # Build independent pivot metrics for this block
        pivot_s1 = combined_block.pivot(index="Y_Label", columns="Protein", values="Delta_State1")
        pivot_s2 = combined_block.pivot(index="Y_Label", columns="Protein", values="Delta_State2")
        
        # Ensure proper depth tracking hierarchy (numbers low-to-high, Baseline dead last)
        custom_order = sorted(
            combined_block["Y_Label"].unique(),
            key=lambda x: (1 if "Baseline" in x else 0, get_numeric_suffix(x)),
        )
        pivot_s1 = pivot_s1.reindex(custom_order)
        pivot_s2 = pivot_s2.reindex(custom_order)

        # Plot State 1 Heatmap
        sns.heatmap(
            pivot_s1, annot=True, fmt=".3f", cmap=cmap, vmin=vmin, vmax=vmax,
            ax=ax_s1, cbar=False, linewidths=0.5, annot_kws={"size": 9}
        )
        ax_s1.set_ylabel(exp_type, fontsize=12, weight="bold")
        
        # Plot State 2 Heatmap
        # Only attach a global color bar legend to the very last panel row to save real estate
        show_cbar = (idx == num_experiments - 1)
        cbar_kwargs = {"label": "Change Relative to Baseline (Δ)", "pad": 0.03} if show_cbar else {}
        
        sns.heatmap(
            pivot_s2, annot=True, fmt=".3f", cmap=cmap, vmin=vmin, vmax=vmax,
            ax=ax_s2, cbar=show_cbar, cbar_kws=cbar_kwargs, linewidths=0.5, annot_kws={"size": 9}
        )
        ax_s2.set_ylabel("")

        # Add metric section titles above only the topmost row panels
        if idx == 0:
            ax_s1.set_title("Δ TM-Score: State 1 (IF/I)", fontsize=13, weight="bold", pad=12)
            ax_s2.set_title("Δ TM-Score: State 2 (OF/A)", fontsize=13, weight="bold", pad=12)

        # Clean up X-Axis labels for bottom-most rows
        if idx == num_experiments - 1:
            ax_s1.set_xlabel("Protein Target", fontsize=11, labelpad=10)
            ax_s2.set_xlabel("Protein Target", fontsize=11, labelpad=10)
            ax_s1.tick_params(axis='x', rotation=45)
            ax_s2.tick_params(axis='x', rotation=45)
        else:
            ax_s1.set_xlabel("")
            ax_s2.set_xlabel("")

    plt.tight_layout()

    # Save output to disk
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    print(f"[Success] Grouped Experiment Heatmap saved to: {output_image_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate structural delta heatmaps grouped by experiment category.")
    parser.add_argument("-i", "--input_md", type=str, required=True, help="Path to your .md file.")
    parser.add_argument("-o", "--output_png", type=str, default="ablation_delta_heatmap.png", help="Output destination image path.")
    args = parser.parse_args()

    try:
        generate_heatmap_from_md(args.input_md, args.output_png)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to plot grouped heatmap: {e}")


if __name__ == "__main__":
    main()