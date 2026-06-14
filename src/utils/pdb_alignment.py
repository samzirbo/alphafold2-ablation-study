"""Reusable PyMOL helpers for aligning PDB structures.

The first PDB is always the fixed reference; the second PDB is transformed onto
that reference. This gives callers a deterministic convention for all
downstream rendering and comparison scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

SaveMode = Literal["first", "second", "both"]
ALIGN_SELECTION = "name CA"


def align_pdb_structures(
    first_pdb: Path,
    second_pdb: Path,
    output_pdb: Path,
    *,
    save: SaveMode = "second",
) -> Path:
    """Align ``second_pdb`` onto ``first_pdb`` and save the requested structure.

    Alignment uses PyMOL ``align`` over C-alpha atoms. The first input is always
    the fixed reference and the second input is always the mobile structure.
    ``save="both"`` writes two MODEL records in the original argument order:
    model 1 is the reference, model 2 is the aligned mobile structure.
    """
    import pymol
    from pymol import cmd

    first_pdb = first_pdb.expanduser().resolve()
    second_pdb = second_pdb.expanduser().resolve()
    output_pdb = output_pdb.expanduser().resolve()

    if not first_pdb.exists():
        raise FileNotFoundError(first_pdb)
    if not second_pdb.exists():
        raise FileNotFoundError(second_pdb)

    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    pymol.finish_launching(["pymol", "-cq"])
    try:
        cmd.load(str(first_pdb), "ref_obj")
        cmd.load(str(second_pdb), "mobile_obj")

        cmd.align(
            f"mobile_obj and ({ALIGN_SELECTION})",
            f"ref_obj and ({ALIGN_SELECTION})",
        )

        if save == "first":
            cmd.save(str(output_pdb), "ref_obj")
        elif save == "second":
            cmd.save(str(output_pdb), "mobile_obj")
        elif save == "both":
            _write_two_model_pdb(cmd, output_pdb)
        else:
            raise ValueError(f"save must be one of first, second, both; got {save!r}")
    finally:
        cmd.quit()

    return output_pdb


def _write_two_model_pdb(cmd, output_pdb: Path) -> None:
    """Write a compact two-model PDB while preserving reference/mobile order."""
    cmd.create("combined", "ref_obj", 1, 1)
    cmd.create("combined", "mobile_obj", 1, 2)

    parts: list[str] = []
    for state, label in ((1, "first_reference"), (2, "second_aligned")):
        parts.append(f"REMARK   1 MODEL {state}: {label}\n")
        parts.append(f"MODEL     {state:4d}\n")
        parts.append(cmd.get_pdbstr("combined", state))
        parts.append("ENDMDL\n")

    output_pdb.write_text("".join(parts))
