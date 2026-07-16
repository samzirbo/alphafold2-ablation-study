import argparse
from pathlib import Path
import io
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import traceback


def get_numeric_suffix(condition_string):
    """
    Extracts numbers safely from a condition string to ensure numerical sorting.
    e.g., 'Lvl 16' -> 16, 'Baseline' -> 0, 'Seed2' -> 2
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
    ABLATION_FONT_SIZE = 10

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
        
        # Format label conditionally: Hide seed if it is Seed0 / Seed 0
        def get_formatted_label(row):
            s_val = str(row[seed_col]).strip()
            if re.match(r'^seed\s*0$', s_val, re.IGNORECASE):
                return str(row["Y_Label"])
            return f"{row['Y_Label']} ({s_val})"

        combined_block["Y_Label_Seed"] = combined_block.apply(get_formatted_label, axis=1)

        # Sort underlying combos
        unique_combos = combined_block[["Y_Label", seed_col]].drop_duplicates()
        sorted_combos = sorted(
            unique_combos.itertuples(index=False),
            key=lambda row: (
                1 if "Baseline" in str(row.Y_Label) else 0,
                get_numeric_suffix(row.Y_Label),
                get_numeric_suffix(getattr(row, seed_col))
            )
        )
        
        # --- GENERATE ORDER & INSERT BLANK ROW PADDING ---
        custom_order = []
        padded_rows = []
        
        prev_group = None
        space_counter = 1  # Unique space strings prevent Pandas from collapsing duplicate blank rows

        for row in sorted_combos:
            current_group = str(row.Y_Label)
            s_val = str(getattr(row, seed_col)).strip()
            
            # Detect group transition (e.g. 15% -> 30%, or 30% -> Baseline)
            if prev_group is not None and prev_group != current_group:
                blank_label = " " * space_counter
                space_counter += 1
                
                custom_order.append(blank_label)
                
                # Add an empty dummy row to the dataset with NaN values
                for protein in combined_block["Protein"].unique():
                    padded_rows.append({
                        "Protein": protein,
                        "Y_Label_Seed": blank_label,
                        "Delta_State1": np.nan,
                        "Delta_State2": np.nan
                    })
            
            # Append current actual item
            if re.match(r'^seed\s*0$', s_val, re.IGNORECASE):
                lbl = current_group
            else:
                lbl = f"{current_group} ({s_val})"
                
            custom_order.append(lbl)
            prev_group = current_group

        if padded_rows:
            padding_df = pd.DataFrame(padded_rows)
            combined_block = pd.concat([combined_block, padding_df], ignore_index=True)

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

        # --- PROCESS SEPARATELY FOR EACH STATE ---
        for state_key, cfg in states_config.items():
            # Pivot, reindex using our padded sorting order
            pivot_data = combined_block.pivot_index = combined_block.pivot_table(
                index="Y_Label_Seed", columns="Protein", values=cfg["val_col"], dropna=False
            )
            pivot_data = pivot_data.reindex(custom_order)
            
            fig_height = max(5, len(custom_order) * 0.45)
            fig, ax = plt.subplots(figsize=(10, fig_height))
            
            sns.heatmap(
                pivot_data, annot=show_annotations, fmt=".3f", cmap=cmap, vmin=vmin, vmax=vmax,
                ax=ax, cbar=True, cbar_kws={"label": "Absolute Change to Baseline (Δ)"},
                linewidths=0.5, annot_kws={"size": 9}
            )
            
            full_title = f"{cfg['title']}"
            if len(unique_seeds) > 1:
                full_title += " (All Seeds Unified)"
            
            ax.set_ylabel(f"{exp_type} & Seed", fontsize=Y_AXIS_FONT_SIZE, weight="bold")
            ax.set_xlabel("Protein Target", fontsize=X_AXIS_FONT_SIZE, labelpad=10)
            ax.set_title(full_title, fontsize=TITLE_FONT_SIZE, weight="bold", pad=12)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=PROTEIN_FONT_SIZE)
            plt.setp(ax.get_yticklabels(), fontsize=ABLATION_FONT_SIZE)

            state_output_path = output_path_obj.parent / f"{output_path_obj.stem}_{cfg['suffix']}{output_path_obj.suffix}"
            
            plt.tight_layout()
            plt.savefig(state_output_path, dpi=300, bbox_inches="tight")
            print(f"[Success] Saved consolidated padded graphic: {state_output_path}")
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