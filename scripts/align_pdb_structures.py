"""Align two PDB structures with PyMOL and save one or both structures.

Examples:
    # Align second.pdb onto first.pdb and save the aligned second structure
    scripts/align_pdb_structures.py first.pdb second.pdb aligned_second.pdb

    # Save reference + aligned mobile as MODEL 1 / MODEL 2
    scripts/align_pdb_structures.py first.pdb second.pdb pair.pdb --save both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.pdb_alignment import align_pdb_structures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("first_pdb", type=Path, help="Reference/fixed PDB")
    parser.add_argument("second_pdb", type=Path, help="Mobile PDB aligned onto first_pdb")
    parser.add_argument("output_pdb", type=Path, help="Output PDB path")
    parser.add_argument(
        "--save",
        choices=["first", "second", "both"],
        default="second",
        help="Which aligned structure(s) to save. Default: second.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_pdb = align_pdb_structures(
        args.first_pdb,
        args.second_pdb,
        args.output_pdb,
        save=args.save,
    )
    print(f"Aligned second onto first and saved: {output_pdb}")


if __name__ == "__main__":
    main()
