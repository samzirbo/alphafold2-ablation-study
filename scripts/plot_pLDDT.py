import os

import nbformat as nbf
from pathlib import Path

def generate_notebook(path_to_plot_folder, path_to_new_notebook):
    pdb_files = list(Path(path_to_plot_folder).rglob("*.pdb"))
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_code_cell(f"""
from google.colab import drive
drive.mount('/content/drive')
! pip install py3Dmol colabfold   
import py3Dmol
import time
from colabfold.colabfold import plot_plddt_legend
    """))

    for pdb in pdb_files:
        cell1 = nbf.v4.new_markdown_cell(f"""
### Filepath: `{pdb}`
""")

        # Cell 1: view structure
        cell2 = nbf.v4.new_code_cell(f"""
with open("{pdb.as_posix()}") as f:
    pdb_data = f.read()

view = py3Dmol.view(width=1000, height=1000)
view.addModel(pdb_data, "pdb")
view.setStyle({{'cartoon': {{'color': 'spectrum'}}}})
view.zoomTo()
view.show()
plot_plddt_legend().show()
    """)

        cells.extend([cell1, cell2])

    nb["cells"] = cells

    with open(path_to_new_notebook, "w") as f:
        nbf.write(nb, f)
        print("Notebook saved to:", path_to_new_notebook)