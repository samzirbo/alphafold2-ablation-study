import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import re
import os
import matplotlib.cm as cm

# ==========================================
# CONFIGURATION VARIABLES
# ==========================================
# You can run this script without any command-line arguments.
# If you do, the script will use the following default paths.
# To specify the paths, simply change the string values below.
#
# INPUT_MD_PATH: The absolute or relative path to your markdown file input).
#                Windows paths should either use double backslashes (\\),
#                forward slashes (/), or have an 'r' before the string.
#                Example: r"c:\path\to\file.md"
#
# OUTPUT_PNG_PATH: The absolute or relative path to save the generated image.
#                  If set to None or "", it will save to the same directory
#                  as the input file with a "_heatmap.png" suffix.
#                  Example: r"c:\path\to\output.png"
#
# FLIP_COLUMNS_WITH_ROWS: Set to True to transpose the table (swap rows and columns).
#                         Set to False to keep standard orientation.
#
# EXCLUDE_MIN_MAX_MEAN: Set to True to completely drop all columns containing 
#                       "Min", "Max", or "Mean" (case-insensitive) from the table.
# ==========================================
INPUT_MD_PATH = r"c:\Users\franc\OneDrive - TUM\2 - Protein Pred\Code\alphafold2-ablation-study\notebooks\qu_mask0-50_allSeeds_7fix3.md"
OUTPUT_PNG_PATH = None
FLIP_COLUMNS_WITH_ROWS = False
EXCLUDE_MIN_MAX_MEAN = True
# ==========================================

def parse_markdown_table(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    table_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith('|') and line.endswith('|'):
            # Skip the alignment row
            if re.match(r'^\|[\s\-\:]+\|$', line):
                continue
            table_lines.append(line)
            
    if not table_lines:
        raise ValueError("No markdown table found in the file.")
        
    data = []
    headers = []
    for i, line in enumerate(table_lines):
        line = line.strip('|')
        cells = [cell.strip() for cell in line.split('|')]
        if i == 0:
            headers = cells
        else:
            data.append(cells)
            
    df = pd.DataFrame(data, columns=headers)
    return df

def get_text_color(bg_color):
    r, g, b = bg_color[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if luminance < 0.5 else 'black'

def create_heatmap(input_md, output_png):
    df = parse_markdown_table(input_md)
    
    # Exclude columns if requested
    if EXCLUDE_MIN_MAX_MEAN:
        cols_to_keep = [c for c in df.columns if not any(x in c.lower() for x in ['min', 'max', 'mean'])]
        df = df[cols_to_keep]
    
    # Identify target cells and build a boolean mask
    mask = np.zeros(df.shape, dtype=bool)
    min_val = 0
    max_val = float('-inf')
    
    for c_idx, c_name in enumerate(df.columns):
        if 'Base State' in c_name:
            for r_idx in range(len(df)):
                try:
                    val = float(df.iloc[r_idx, c_idx])
                    if val >= 0:
                        mask[r_idx, c_idx] = True
                        if val > max_val: 
                            max_val = val
                except ValueError:
                    pass
                    
    if max_val == float('-inf'):
        max_val = 1
        
    # Transpose if requested
    if FLIP_COLUMNS_WITH_ROWS:
        # Transpose dataframe: columns become index, index becomes columns
        df = df.T.reset_index()
        df.rename(columns={'index': ''}, inplace=True)
        # Ensure new columns are strings
        df.columns = df.columns.astype(str)
        
        # Transpose the mask and pad the first column (which is now the headers) with False
        new_mask = np.zeros(df.shape, dtype=bool)
        new_mask[:, 1:] = mask.T
        mask = new_mask
        
    # Plotting
    if FLIP_COLUMNS_WITH_ROWS:
        fig, ax = plt.subplots(figsize=(df.shape[1] * 1.5 + 2, df.shape[0] * 0.4 + 2))
    else:
        fig, ax = plt.subplots(figsize=(df.shape[1] * 1.8, df.shape[0] * 0.35 + 2))
        
    # Adjust subplot to explicitly leave room for the colorbar at the bottom to prevent overlap
    plt.subplots_adjust(bottom=0.2)
    ax.axis('off')
    
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
                     
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    
    # Style header row
    for c in range(len(df.columns)):
        cell = table[(0, c)]
        cell.set_facecolor('#f2f2f2')
        cell.set_text_props(weight='bold')
        
    if FLIP_COLUMNS_WITH_ROWS:
        # Style the first column as headers too
        for r in range(len(df)):
            cell = table[(r + 1, 0)]
            cell.set_facecolor('#f2f2f2')
            cell.set_text_props(weight='bold')
    
    # Apply colors based on mask
    cmap = plt.get_cmap('coolwarm')
    norm = Normalize(vmin=min_val, vmax=max_val)
    
    for r in range(len(df)):
        for c in range(len(df.columns)):
            if mask[r, c]:
                cell = table[(r + 1, c)]
                val = float(df.iloc[r, c])
                color = cmap(norm(val))
                cell.set_facecolor(color)
                cell.get_text().set_color(get_text_color(color))
                
    # Add a separate axes for the colorbar to prevent any overlap with the table
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Position: [left, bottom, width, height] in figure coordinate space
    cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.03])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Value Range')
    
    # Do not call plt.tight_layout() as it may ruin the custom axes placement
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Heatmap successfully saved to {output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a heatmap from a markdown table.")
    parser.add_argument("input_md", nargs='?', default=INPUT_MD_PATH, help="Path to the input markdown file.")
    parser.add_argument("-o", "--output", help="Path to the output PNG image.", default=OUTPUT_PNG_PATH)
    args = parser.parse_args()
    
    output_path = args.output
    if not output_path:
        base, _ = os.path.splitext(args.input_md)
        output_path = f"{base}_heatmap.png"
        
    create_heatmap(args.input_md, output_path)
