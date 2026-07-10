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


def generate_heatmap_from_md(md_file_path: str, output_image_path: str = "ablation_delta_heatmap.png", show_annotations: bool = True, vmin: float = -0.2, vmax: float = 0.2):
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
    # max_val = max(df["Delta_State1"].abs().max(), df["Delta_State2"].abs().max())
    # if pd.isna(max_val) or max_val == 0:
    #     max_val = 1.0  
    # vmin, vmax = -max_val, max_val

    output_path_obj = Path(output_image_path)

    # Global Font Size Variables
    TITLE_FONT_SIZE = 16
    X_AXIS_FONT_SIZE = 20  
    Y_AXIS_FONT_SIZE = 20 
    PROTEIN_FONT_SIZE = 15
    ABLATION_FONT_SIZE = 15

    for exp_type in unique_experiments:
        # Extract experiment rows
        exp_data = experiment_df[experiment_df["Experiment Type"] == exp_type].copy()
        if exp_type == "Depth":
            exp_data["Y_Label"] = exp_data["Ablation Level"].astype(int, errors='ignore').astype(str)
        else: 
            exp_data["Y_Label"] = exp_data["Ablation Level"].astype(int, errors='ignore').astype(str) + "%"
        
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

        # Sanitize filename string
        clean_exp_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', exp_type)
        
        # Helper dictionary to dynamically handle both state conditions cleanly
        states_config = {
            "State1": {
                "data": pivot_s1,
                "title": "Δ TM-Score: State 1 (IF/I)",
                "suffix": f"{clean_exp_name}_State1"
            },
            "State2": {
                "data": pivot_s2,
                "title": "Δ TM-Score: State 2 (OF/A)",
                "suffix": f"{clean_exp_name}_State2"
            }
        }

        # --- PROCESS AND GENERATE EACH GRAPH SEPARATELY ---
        for state_key, cfg in states_config.items():
            # Standard independent figure dimensions (scaled nicely for 1 matrix + colorbar)
            fig, ax = plt.subplots(figsize=(8.5, 5))
            
            # Draw heatmap (both get their own colorbar now since they are solo graphs)
            cbar_kwargs = {"label": "Absolute Change to Baseline (Δ)", "pad": 0.03}
            sns.heatmap(
                cfg["data"], annot=show_annotations, fmt=".3f", cmap=cmap, vmin=vmin, vmax=vmax,
                ax=ax, cbar=True, cbar_kws=cbar_kwargs, linewidths=0.5, annot_kws={"size": 9}
            )
            
            # Format Colorbar text
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label("Absolute Change to Baseline (Δ)", fontsize=14, weight="bold")

            # Structural Label Adjustments
            ax.set_ylabel(exp_type, fontsize=Y_AXIS_FONT_SIZE, weight="bold")
            ax.set_xlabel("Protein Target", fontsize=X_AXIS_FONT_SIZE, labelpad=10)
            ax.set_title(cfg["title"], fontsize=TITLE_FONT_SIZE, weight="bold", pad=12)
            
            # Tick Text Properties
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=PROTEIN_FONT_SIZE)
            plt.setp(ax.get_yticklabels(), fontsize=ABLATION_FONT_SIZE)

            # Build individual path (e.g. baseline_heatmap_Mutation_X_State1.png)
            state_output_path = output_path_obj.parent / f"{output_path_obj.stem}_{cfg['suffix']}{output_path_obj.suffix}"
            
            plt.tight_layout()
            plt.savefig(state_output_path, dpi=300, bbox_inches="tight")
            print(f"[Success] Saved standalone state graphic to: {state_output_path}")
            
            # Drop current figure from execution stack memory
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate structural delta heatmaps grouped by experiment category.")
    parser.add_argument("-i", "--input_md", type=str, required=True, help="Path to your .md file.")
    parser.add_argument("-o", "--output_png", type=str, default="ablation_delta_heatmap.png", help="Output destination image path.")
    parser.add_argument("-sa", "--show_annot", action="store_false", help="Show numerical text annotations inside the heatmap cells.")
    
    parser.add_argument("--vmin", type=float, default=-0.2, help="Fixed minimum value for heatmap color scale.")
    parser.add_argument("--vmax", type=float, default=0.2, help="Fixed maximum value for heatmap color scale.")
    
    args = parser.parse_args()

    try:
        generate_heatmap_from_md(args.input_md, args.output_png, show_annotations = not args.show_annot, vmin=args.vmin, 
            vmax=args.vmax)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to plot grouped heatmap: {e}")


if __name__ == "__main__":
    main()