import argparse
from pathlib import Path
import io
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import traceback


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
            
    # Standardize column parsing
    df["Ablation Level"] = pd.to_numeric(df["Ablation Level"], errors='coerce')
    
    # Check if 'Seed' column exists
    seed_col = None
    for col in df.columns:
        if col.lower() == 'seed':
            seed_col = col
            break
            
    if seed_col is None:
        print("[Info] No 'Seed' column detected. Visualizing as single seed dataset.")
        df["Seed"] = "Seed0"
        seed_col = "Seed"

    # Identify dynamic column names for Delta States
    s1_col = [c for c in df.columns if "Δ Base" in c and any(sub in c for sub in ["State 1", "Inward", "Inactive"])][0]
    s2_col = [c for c in df.columns if "Δ Base" in c and any(sub in c for sub in ["State 2", "Outward", "Active"])][0]

    df["Delta_State1"] = pd.to_numeric(df[s1_col], errors='coerce')
    df["Delta_State2"] = pd.to_numeric(df[s2_col], errors='coerce')

    baseline_mask = df["Experiment Type"].str.lower() == "baseline"
    baseline_df = df[baseline_mask].copy()
    experiment_df = df[~baseline_mask].copy()
    
    unique_experiments = sorted(experiment_df["Experiment Type"].unique())

    if not unique_experiments:
        raise ValueError("No distinct experiment types found outside of Baseline rows.")

    cmap = "coolwarm"  
    output_path_obj = Path(output_image_path)

    # Styling constants
    TITLE_FONT_SIZE = 14
    X_AXIS_FONT_SIZE = 14  
    Y_AXIS_FONT_SIZE = 14 
    PROTEIN_FONT_SIZE = 11
    ABLATION_FONT_SIZE = 11

    for exp_type in unique_experiments:
        # Extract specific experiment data
        exp_data = experiment_df[experiment_df["Experiment Type"] == exp_type].copy()
        
        # Determine labels for rows
        exp_data["Y_Label"] = exp_data["Ablation Level"].astype(int, errors='ignore').astype(str)
        if exp_type != "Depth":
            exp_data["Y_Label"] = exp_data["Y_Label"] + "%"
        
        # Replicate baseline metrics across all unique seeds present in exp_data
        unique_seeds = exp_data[seed_col].unique()
        baseline_replicated = []
        for s in unique_seeds:
            b_temp = baseline_df.copy()
            b_temp["Experiment Type"] = exp_type
            b_temp["Y_Label"] = "Baseline"
            b_temp[seed_col] = s
            baseline_replicated.append(b_temp)
            
        if baseline_replicated:
            current_baseline = pd.concat(baseline_replicated, ignore_index=True)
        else:
            current_baseline = pd.DataFrame()
        
        combined_block = pd.concat([exp_data, current_baseline], ignore_index=True)
        
        # Build clean numerical index ordering
        custom_order = sorted(
            combined_block["Y_Label"].unique(),
            key=lambda x: (1 if "Baseline" in str(x) else 0, get_numeric_suffix(x)),
        )

        states_config = {
            "State1": {
                "val_col": "Delta_State1",
                "title": f"Δ TM-Score: State 1 (IF/I)",
                "suffix": f"{re.sub(r'[^a-zA-Z0-9_\-]', '_', exp_type)}_State1"
            },
            "State2": {
                "val_col": "Delta_State2",
                "title": f"Δ TM-Score: State 2 (OF/A)",
                "suffix": f"{re.sub(r'[^a-zA-Z0-9_\-]', '_', exp_type)}_State2"
            }
        }

        # --- PROCESS SEPARATELY FOR EACH STATE AND SEED ---
        sorted_seeds = sorted(list(unique_seeds), key=lambda x: get_numeric_suffix(x))

        for state_key, cfg in states_config.items():
            for seed in sorted_seeds:
                # Isolate data for this specific seed
                seed_block = combined_block[combined_block[seed_col] == seed]
                
                # Pivot and reindex to align structured layout
                pivot_data = seed_block.pivot(index="Y_Label", columns="Protein", values=cfg["val_col"])
                pivot_data = pivot_data.reindex(custom_order)
                
                # Create standard single figure layout
                fig, ax = plt.subplots(figsize=(8.5, 5))
                
                sns.heatmap(
                    pivot_data, annot=show_annotations, fmt=".3f", cmap=cmap, vmin=vmin, vmax=vmax,
                    ax=ax, cbar=True, cbar_kws={"label": "Absolute Change to Baseline (Δ)"},
                    linewidths=0.5, annot_kws={"size": 9}
                )
                
                # Add seed information to the title dynamically
                full_title = f"{cfg['title']} ({seed})"
                
                ax.set_ylabel(exp_type, fontsize=Y_AXIS_FONT_SIZE, weight="bold")
                ax.set_xlabel("Protein Target", fontsize=X_AXIS_FONT_SIZE, labelpad=10)
                ax.set_title(full_title, fontsize=TITLE_FONT_SIZE, weight="bold", pad=12)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=PROTEIN_FONT_SIZE)
                plt.setp(ax.get_yticklabels(), fontsize=ABLATION_FONT_SIZE)

                # Append seed to filename dynamically (e.g. baseline_heatmap_Mutation_X_State1_Seed1.png)
                state_output_path = output_path_obj.parent / f"{output_path_obj.stem}_{cfg['suffix']}_{seed}{output_path_obj.suffix}"
                
                plt.tight_layout()
                plt.savefig(state_output_path, dpi=300, bbox_inches="tight")
                print(f"[Success] Saved standalone graphic: {state_output_path}")
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
        generate_heatmap_from_md(
            args.input_md, args.output_png, show_annotations=not args.show_annot, 
            vmin=args.vmin, vmax=args.vmax
        )
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to plot grouped heatmap: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()