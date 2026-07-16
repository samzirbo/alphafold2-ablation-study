import argparse
from pathlib import Path
import io
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
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


def draw_diagonal_split_heatmap(pivot_s1, pivot_s2, custom_order, proteins, cmap_name, vmin, vmax, show_annotations, title, ylabel, xlabel, output_path, figsize, fonts_config):
    """
    Renders a custom heatmap where each cell is divided diagonally into two triangles.
    Styled to match Seaborn's default heatmap properties perfectly.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Establish grid parameters
    nrows = len(custom_order)
    ncols = len(proteins)
    
    # Set limits and background color matching Seaborn's empty/NaN look
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_facecolor("#f0f0f0") 
    
    # Get color mapping
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    patches_s1, colors_s1 = [], []
    patches_s2, colors_s2 = [], []
    
    for r_idx, row_label in enumerate(custom_order):
        # Identify padding/empty rows
        if not row_label.strip():
            continue
            
        for c_idx, col_label in enumerate(proteins):
            val_s1 = pivot_s1.at[row_label, col_label]
            val_s2 = pivot_s2.at[row_label, col_label]
            
            if pd.isna(val_s1) or pd.isna(val_s2):
                continue
                
            x0, x1 = c_idx, c_idx + 1
            y0, y1 = nrows - 1 - r_idx, nrows - r_idx  # Invert Y to match pandas top-down rendering
            
            # Triangle 1: Upper-Left (State 1)
            t1 = Polygon([[x0, y1], [x1, y1], [x0, y0]], closed=True)
            patches_s1.append(t1)
            colors_s1.append(cmap(norm(val_s1)))
            
            # Triangle 2: Lower-Right (State 2)
            t2 = Polygon([[x1, y1], [x1, y0], [x0, y0]], closed=True)
            patches_s2.append(t2)
            colors_s2.append(cmap(norm(val_s2)))
            
            if show_annotations:
                # Text color logic matching Seaborn's color thresholding
                c_s1 = "white" if abs(val_s1) > (vmax - vmin) * 0.3 else "black"
                c_s2 = "white" if abs(val_s2) > (vmax - vmin) * 0.3 else "black"
                
                ax.text(x0 + 0.30, y0 + 0.70, f"{val_s1:.3f}", 
                        color=c_s1, ha="center", va="center", fontsize=8, weight="normal")
                ax.text(x0 + 0.70, y0 + 0.30, f"{val_s2:.3f}", 
                        color=c_s2, ha="center", va="center", fontsize=8, weight="normal")

    # Add polygon collections with clean white borders matching Seaborn's linewidths=0.5
    p_coll1 = PatchCollection(patches_s1, facecolors=colors_s1, edgecolors='white', linewidths=0.5)
    p_coll2 = PatchCollection(patches_s2, facecolors=colors_s2, edgecolors='white', linewidths=0.5)
    ax.add_collection(p_coll1)
    ax.add_collection(p_coll2)
    
    # Configure exact styling matching Seaborn tick parameters
    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_xticklabels(proteins, rotation=45, ha="right", fontsize=fonts_config["PROTEIN_FONT_SIZE"])
    
    # Align row labels matching top-down reindexed ordering
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_yticklabels(list(reversed(custom_order)), fontsize=fonts_config["ABLATION_FONT_SIZE"])
    
    # Remove outer spines/borders & tick ticks (Seaborn default heatmap behavior)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)
    
    # Labels and Titles
    ax.set_ylabel(ylabel, fontsize=fonts_config["Y_AXIS_FONT_SIZE"], weight="bold")
    ax.set_xlabel(xlabel, fontsize=fonts_config["X_AXIS_FONT_SIZE"], labelpad=10)
    ax.set_title(title, fontsize=fonts_config["TITLE_FONT_SIZE"], weight="bold", pad=12)
    
    # Colorbar configuration matching Seaborn
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.04, aspect=20)
    cbar.set_label("Absolute Change to Baseline (Δ)", fontsize=11)
    cbar.outline.set_visible(False)  # Match Seaborn flat colorbar style
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[Success] Saved consolidated diagonal-split graphic: {output_path}")
    plt.close(fig)


def generate_heatmap_from_md(md_file_path: str, output_image_path: str = "ablation_delta_heatmap.png", 
                             show_annotations: bool = True, vmin: float = -0.2, vmax: float = 0.2, 
                             state_combine: bool = False):
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
        df["Seed"] = "Default"
        seed_col = "Seed"

# Identify dynamic column names for Delta States (updated for robust matching)
    s1_col = [
        c for c in df.columns 
        if "Δ Base" in c and any(sub in c for sub in ["State 1", "IF", "IF/I", "Inward", "Inactive"])
    ][0]
    
    s2_col = [
        c for c in df.columns 
        if "Δ Base" in c and any(sub in c for sub in ["State 2", "OF", "OF/A", "Outward", "Active"])
    ][0]





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

    # Styling constants bundled into a single configuration block
    fonts_config = {
        "TITLE_FONT_SIZE": 14,
        "X_AXIS_FONT_SIZE": 14,
        "Y_AXIS_FONT_SIZE": 14,
        "PROTEIN_FONT_SIZE": 11,
        "ABLATION_FONT_SIZE": 10
    }

    for exp_type in unique_experiments:
        exp_data = experiment_df[experiment_df["Experiment Type"] == exp_type].copy()
        
        exp_data["Y_Label"] = exp_data["Ablation Level"].astype(int, errors='ignore').astype(str)
        if exp_type != "Depth":
            exp_data["Y_Label"] = exp_data["Y_Label"] + "%"
        
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
        
        def get_formatted_label(row):
            s_val = str(row[seed_col]).strip()
            if len(unique_seeds) <= 1 or re.match(r'^(default)$', s_val, re.IGNORECASE):
                return str(row["Y_Label"])
            return f"{row['Y_Label']} ({s_val})"

        combined_block["Y_Label_Seed"] = combined_block.apply(get_formatted_label, axis=1)

        unique_combos = combined_block[["Y_Label", seed_col]].drop_duplicates()
        sorted_combos = sorted(
            unique_combos.itertuples(index=False),
            key=lambda row: (
                1 if "Baseline" in str(row.Y_Label) else 0,
                get_numeric_suffix(row.Y_Label),
                get_numeric_suffix(getattr(row, seed_col))
            )
        )
        
        custom_order = []
        padded_rows = []
        prev_group = None
        space_counter = 1

        for row in sorted_combos:
            current_group = str(row.Y_Label)
            s_val = str(getattr(row, seed_col)).strip()
            
            # ONLY add blank padding rows if we actually have multiple seeds to separate
            if len(unique_seeds) > 1 and prev_group is not None and prev_group != current_group:
                blank_label = " " * space_counter
                space_counter += 1
                
                custom_order.append(blank_label)
                
                for protein in combined_block["Protein"].unique():
                    padded_rows.append({
                        "Protein": protein,
                        "Y_Label_Seed": blank_label,
                        "Delta_State1": np.nan,
                        "Delta_State2": np.nan
                    })
            
            if len(unique_seeds) <= 1 or re.match(r'^(default)$', s_val, re.IGNORECASE):
                lbl = current_group
            else:
                lbl = f"{current_group} ({s_val})"
                
            custom_order.append(lbl)
            prev_group = current_group

        if padded_rows:
            padding_df = pd.DataFrame(padded_rows)
            combined_block = pd.concat([combined_block, padding_df], ignore_index=True)

        proteins_list = sorted(list(combined_block["Protein"].dropna().unique()))

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

        # --- OPTION 1: DIAGONAL SPLIT COMBINED HEATMAP ---
        if state_combine:
            pivot_s1 = combined_block.pivot_table(index="Y_Label_Seed", columns="Protein", values="Delta_State1", dropna=False)
            pivot_s2 = combined_block.pivot_table(index="Y_Label_Seed", columns="Protein", values="Delta_State2", dropna=False)
            
            pivot_s1 = pivot_s1.reindex(custom_order)
            pivot_s2 = pivot_s2.reindex(custom_order)
            
            fig_height = max(5, len(custom_order) * 0.45)
            
            title = f"Δ TM-Score: States 1 & 2 Combined"
            if len(unique_seeds) > 1:
                #title += " (All Seeds Unified)"
                title = title
                
            state_output_path = output_path_obj.parent / f"{output_path_obj.stem}_{re.sub(r'[^a-zA-Z0-9_\-]', '_', exp_type)}_Combined{output_path_obj.suffix}"
            
            draw_diagonal_split_heatmap(
                pivot_s1=pivot_s1, pivot_s2=pivot_s2, custom_order=custom_order, proteins=proteins_list,
                cmap_name=cmap, vmin=vmin, vmax=vmax, show_annotations=show_annotations,
                title=title, ylabel=f"{exp_type}", xlabel="Protein Target",
                output_path=state_output_path, figsize=(10, fig_height), fonts_config=fonts_config
            )

        # --- OPTION 2: DEFAULT SEPARATED HEATMAPS ---
        else:
            for state_key, cfg in states_config.items():
                pivot_data = combined_block.pivot_table(
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
                    #full_title += " (All Seeds Unified)"
                    full_title = full_title
                
                ax.set_ylabel(f"{exp_type}", fontsize=fonts_config["Y_AXIS_FONT_SIZE"], weight="bold")
                ax.set_xlabel("Protein Target", fontsize=fonts_config["X_AXIS_FONT_SIZE"], labelpad=10)
                ax.set_title(full_title, fontsize=fonts_config["TITLE_FONT_SIZE"], weight="bold", pad=12)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fonts_config["PROTEIN_FONT_SIZE"])
                plt.setp(ax.get_yticklabels(), fontsize=fonts_config["ABLATION_FONT_SIZE"])

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
    parser.add_argument("--state-combine", action="store_true", help="Combine State 1 and State 2 into diagonally split cells on a single plot.")
    
    args = parser.parse_args()

    try:
        generate_heatmap_from_md(
            args.input_md, args.output_png, show_annotations=not args.show_annot, 
            vmin=args.vmin, vmax=args.vmax, state_combine=args.state_combine
        )
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to plot grouped heatmap: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()