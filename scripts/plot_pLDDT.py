import os

import nbformat as nbf
from pathlib import Path

def generate_notebook(path_to_plot_folder):
    pdb_files = list(Path(path_to_plot_folder).rglob("*.pdb"))
    print(path_to_plot_folder)
    print(pdb_files)
    pdb_files = list(Path(path_to_plot_folder).glob("*.pdb"))

    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_code_cell(f"""
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

view = py3Dmol.view(width=400, height=400)
view.addModel(pdb_data, "pdb")
view.setStyle({{'cartoon': {{'color': 'spectrum'}}}})
view.zoomTo()
view.show()
plot_plddt_legend().show()
    """)

        # Cell 2: save image
        cell3 = nbf.v4.new_code_cell(f"""
time.sleep(4)
png = view.png()
    """)

        cells.extend([cell1, cell2, cell3])

    nb["cells"] = cells

    with open("plddt_batch_render.ipynb", "w") as f:
        nbf.write(nb, f)
        print("Saved to:", os.getcwd() + "/plddt_batch_render.ipynb")