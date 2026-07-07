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


def generate_heatmap_from_md(md_file_path: str, output_image_path: str = "ablation_delta_heatmap.png", show_annotations: bool = True):
    path = Path(md_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find the specified Markdown report at: {md_file_path}")
        
    print(f"[Info] Scanning file line-by-line for Markdown tables: {md_file_path}")
    
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
    
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(how="all", axis=0)
    
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()
            
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

    baseline_mask = df["Experiment Type"].str.lower() == "baseline"
    baseline_df = df[baseline_mask].copy()
    experiment_df = df[~baseline_mask].copy()
    
    unique_experiments = sorted(experiment_df["Experiment Type"].unique())

    if not unique_experiments:
        raise ValueError("No distinct experiment types found outside of Baseline rows.")

    cmap = "coolwarm"  

    # Establish global symmetric bounds centered at 0.0
    max_val = max(df["Delta_State1"].abs().max(), df["Delta_State2"].abs().max())
    if pd.isna(max_val) or max_val == 0:
        max_val = 1.0  
    vmin, vmax = -max_val, max_val

    # base output name components to dynamically append experiment names
    output_path_obj = Path(output_image_path)

    # --- THE CRITICAL LOOP CHANGE ---
    for exp_type in unique_experiments:
        # Create a brand new figure for EACH individual experiment (1 row, 2 columns)
        fig, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(14, 4.5))
        
        # Extract experiment rows
        exp_data = experiment_df[experiment_df["Experiment Type"] == exp_type].copy()
        if exp_type=="Depth":
            exp_data["Y_Label"] = exp_data["Ablation Level"].astype(int, errors='ignore').astype(str)
        else: 
            exp_data["Y_Label"] = exp_data["Ablation Level"].astype(int, errors='ignore').astype(str)+ "%"
        
        # Mirror the baseline records
        current_baseline = baseline_df.copy()
        current_baseline["Experiment Type"] = exp_type
        current_baseline["Y_Label"] = "Baseline"
        
        combined_block = pd.concat([exp_data, current_baseline], ignore_index=True)
        
        pivot_s1 = combined_block.pivot_index_or_values(index="Y_Label", columns="Protein", values="Delta_State1") if hasattr(combined_block, 'pivot_index_or_values') else combined_block.pivot(index="Y_Label", columns="Protein", values="Delta_State1")
        pivot_s2 = combined_block.pivot(index="Y_Label", columns="Protein", values="Delta_State2")
        
        custom_order = sorted(
            combined_block["Y_Label"].unique(),
            key=lambda x: (1 if "Baseline" in str(x) else 0, get_numeric_suffix(x)),
        )
        pivot_s1 = pivot_s1.reindex(custom_order)
        pivot_s2 = pivot_s2.reindex(custom_order)

        TITLE_FONT_SIZE = 16
        X_AXIS_FONT_SIZE = 20  # Target Protein
        Y_AXIS_FONT_SIZE = 20 # like Col Mask 
        PROTEIN_FONT_SIZE = 15
        ABLATION_FONT_SIZE = 15


        # Plot State 1 Matrix
        sns.heatmap(
            pivot_s1, annot=show_annotations, fmt=".3f", cmap=cmap, vmin=vmin, vmax=vmax,
            ax=ax_s1, cbar=False, linewidths=0.5, annot_kws={"size": 9}
        )
        ax_s1.set_ylabel(exp_type, fontsize=Y_AXIS_FONT_SIZE, weight="bold")
        
        # Plot State 2 Matrix (Every individual plot now gets its own colorbar)
        cbar_kwargs = {"label": "Change Relative to Baseline (Δ)", "pad": 0.03}
        sns.heatmap(
            pivot_s2, annot=show_annotations, fmt=".3f", cmap=cmap, vmin=vmin, vmax=vmax,
            ax=ax_s2, cbar=True, cbar_kws=cbar_kwargs, linewidths=0.5, annot_kws={"size": 9}
        )
        # --- ADJUST LEGEND FONT SIZES HERE ---
        cbar = ax_s2.collections[0].colorbar  # Access the generated colorbar object

        cbar.ax.tick_params(labelsize=12)      # <-- Increase size of the scale numbers (-0.4, 0.0, 0.4, etc.)
        cbar.set_label("Change Relative to Baseline (Δ)", fontsize=14, weight="bold")


        ax_s2.set_ylabel("")

        # Titles are now added to every single generated image
        ax_s1.set_title(f"Δ TM-Score: State 1 (IF/I)", fontsize=TITLE_FONT_SIZE , weight="bold", pad=12)
        ax_s2.set_title(f"Δ TM-Score: State 2 (OF/A)", fontsize=TITLE_FONT_SIZE , weight="bold", pad=12)

        # Every plot is the bottom plot now, so always show X-axis labels properly
        ax_s1.set_xlabel("Protein Target", fontsize=X_AXIS_FONT_SIZE, labelpad=10)
        ax_s2.set_xlabel("Protein Target", fontsize=X_AXIS_FONT_SIZE, labelpad=10)
        plt.setp(ax_s1.get_xticklabels(), rotation=45, ha="right", fontsize= PROTEIN_FONT_SIZE)
        plt.setp(ax_s2.get_xticklabels(), rotation=45, ha="right", fontsize = PROTEIN_FONT_SIZE)


        plt.setp(ax_s1.get_yticklabels(), fontsize=ABLATION_FONT_SIZE)
        plt.setp(ax_s2.get_yticklabels(), fontsize=ABLATION_FONT_SIZE)

        # Generate a unique file name for this specific experiment group
        # e.g., "ablation_delta_heatmap.png" becomes "ablation_delta_heatmap_Mutation_X.png"
        clean_exp_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', exp_type) # sanitize filename string
        individual_output_path = output_path_obj.parent / f"{output_path_obj.stem}_{clean_exp_name}{output_path_obj.suffix}"

        plt.tight_layout()
        plt.savefig(individual_output_path, dpi=300, bbox_inches="tight")
        print(f"[Success] Saved individual graphic to: {individual_output_path}")
        
        # Close the plot to free memory before starting the next loop cycle
        plt.close(fig)


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