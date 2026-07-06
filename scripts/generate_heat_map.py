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

    # Combine tracking details into Y-Axis condition label
    df["Condition"] = df["Experiment Type"] + " (Lvl " + df["Ablation Level"].astype(str) + ")"

    df.rename(
        columns={
            "Δ Base State 1 (IF/I)": "Delta_State1",
            "Δ Base State 2 (OF/A)": "Delta_State2",
        },
        inplace=True,
    )

    # Pivot tracking frames
    pivot_s1 = df.pivot(index="Condition", columns="Protein", values="Delta_State1")
    pivot_s2 = df.pivot(index="Condition", columns="Protein", values="Delta_State2")

    # FIX: Re-order index rows to push Baseline to the bottom
    # Groups numeric depths first (0), pushes Baseline last (1), sorting depths internally.
    custom_order = sorted(
        df["Condition"].unique(),
        key=lambda x: (1 if "Baseline" in x else 0, get_numeric_suffix(x)),
    )
    
    pivot_s1 = pivot_s1.reindex(custom_order)
    pivot_s2 = pivot_s2.reindex(custom_order)

    # Initialize Plotting Canvas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    cmap = "coolwarm"  

    # Balance diverging limits perfectly around delta center point 0.0
    max_val = max(abs(df["Delta_State1"].max()), abs(df["Delta_State2"].max()))
    vmin, vmax = -max_val, max_val

    # Subplot 1: State 1
    sns.heatmap(
        pivot_s1,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax1,
        cbar=False,
        linewidths=0.5,
        annot_kws={"size": 9}
    )
    ax1.set_title("Δ TM-Score: State 1 (IF/I)", fontsize=13, weight="bold", pad=12)
    ax1.set_xlabel("Protein Target", fontsize=11, labelpad=10)
    ax1.set_ylabel("Experiment Condition", fontsize=11)
    ax1.tick_params(axis='x', rotation=45)

    # Subplot 2: State 2
    sns.heatmap(
        pivot_s2,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax2,
        cbar_kws={"label": "Change Relative to Baseline (Δ)", "pad": 0.03},
        linewidths=0.5,
        annot_kws={"size": 9}
    )
    ax2.set_title("Δ TM-Score: State 2 (OF/A)", fontsize=13, weight="bold", pad=12)
    ax2.set_xlabel("Protein Target", fontsize=11, labelpad=10)
    ax2.set_ylabel("") 
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save output to disk
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    print(f"[Success] Heatmap plot (Baseline at bottom) saved to: {output_image_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate structural delta heatmaps from a Markdown table report.")
    parser.add_argument("-i", "--input_md", type=str, required=True, help="Path to your .md file.")
    parser.add_argument("-o", "--output_png", type=str, default="ablation_delta_heatmap.png", help="Output destination image path.")
    args = parser.parse_args()

    try:
        generate_heatmap_from_md(args.input_md, args.output_png)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to plot heatmap: {e}")


if __name__ == "__main__":
    main()