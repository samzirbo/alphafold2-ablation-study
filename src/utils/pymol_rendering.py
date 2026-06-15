"""Shared PyMOL renderer for aligned protein snapshots.

The renderer owns the project-specific choices: align to a fixed reference over
C-alpha atoms, color residues with the Protein Viewer sequence-id palette, frame
the helical core consistently per protein, and make GIFs with a slow rocking
motion. PyMOL and imageio imports stay lazy so non-rendering code can import this
module outside the PyMOL environment.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Sequence
from pathlib import Path

# Mol*/Protein Viewer turbo colors used for sequence-id coloring.
MOLSTAR_TURBO_STOPS = [
    (0x4A, 0x41, 0xB5),
    (0x4A, 0x58, 0xDD),
    (0x42, 0x6F, 0xF2),
    (0x39, 0x87, 0xF9),
    (0x2F, 0x9D, 0xF5),
    (0x28, 0xB2, 0xE9),
    (0x25, 0xC6, 0xD8),
    (0x27, 0xD7, 0xC4),
    (0x2E, 0xE5, 0xAE),
    (0x3B, 0xF0, 0x98),
    (0x4D, 0xF8, 0x84),
    (0x62, 0xFD, 0x70),
    (0x7B, 0xFE, 0x5F),
    (0x95, 0xFB, 0x51),
    (0xAF, 0xF4, 0x44),
    (0xC8, 0xEA, 0x3A),
    (0xDE, 0xDD, 0x32),
    (0xF0, 0xCC, 0x2C),
    (0xFE, 0xB9, 0x27),
    (0xFF, 0xA4, 0x23),
    (0xFF, 0x8E, 0x1F),
    (0xFF, 0x76, 0x1C),
    (0xF6, 0x5F, 0x18),
    (0xE5, 0x48, 0x13),
    (0xD0, 0x33, 0x0E),
    (0xBA, 0x22, 0x08),
    (0xA5, 0x14, 0x03),
    (0x96, 0x0D, 0x00),
]
PALETTE_SIZE = 256
ALIGN_SELECTION = "name CA"
ALIGN_CYCLES = 0
ROCKING_ANGLE = 18.0


def interpolate_color(stops: list[tuple[int, int, int]], t: float) -> tuple[float, float, float]:
    """Return an RGB PyMOL color at a normalized position in a discrete palette."""
    if t <= 0:
        return tuple(channel / 255 for channel in stops[0])
    if t >= 1:
        return tuple(channel / 255 for channel in stops[-1])

    scaled = t * (len(stops) - 1)
    left = int(scaled)
    frac = scaled - left
    return tuple(
        (start + (end - start) * frac) / 255
        for start, end in zip(stops[left], stops[left + 1], strict=True)
    )


class PymolRenderer:
    """Reusable PyMOL session for efficient batch rendering."""

    def __init__(
        self,
        *,
        width: int = 1200,
        height: int = 1200,
        dpi: int = 250,
        zoom_buffer: float = 8.0,
    ):
        import pymol
        from pymol import cmd

        self.cmd = cmd
        self.width = width
        self.height = height
        self.dpi = dpi
        self.zoom_buffer = zoom_buffer
        self._reference_pdb: Path | None = None
        self._reference_view = None
        self._prepared_structures: dict[Path, str] = {}

        pymol.finish_launching(["pymol", "-cq"])
        self._define_colors()

    def __enter__(self) -> "PymolRenderer":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self.cmd.quit()

    def prepare_fixed_view(
        self,
        pdb_paths: Sequence[Path],
        *,
        reference_pdb: Path,
    ) -> None:
        """Prepare one stable camera for a related set of aligned structures.

        Hidden objects are aligned once, used to compute the camera bounds, and
        then reused by ``load_structure`` to avoid repeated load/align work.
        """
        reference_pdb = reference_pdb.expanduser().resolve()
        if not reference_pdb.exists():
            raise FileNotFoundError(reference_pdb)

        self._delete_prefixed("view_bounds_")
        self._prepared_structures = {}
        self._load_reference(reference_pdb)

        selections = []
        for i, pdb_path in enumerate(pdb_paths):
            pdb_path = pdb_path.expanduser().resolve()
            if not pdb_path.exists():
                raise FileNotFoundError(pdb_path)
            name = f"view_bounds_{i}"
            self.cmd.load(str(pdb_path), name)
            self._align_to_reference(name)
            selections.append(name)
            self._prepared_structures[pdb_path] = name

        view_selection = self._core_view_selection(selections)
        self.cmd.orient("reference")
        self.cmd.zoom(view_selection, self.zoom_buffer, complete=1)
        self._reference_view = self.cmd.get_view()
        self.cmd.hide("everything", "all")

    def load_structure(
        self,
        pdb_path: Path,
        *,
        reference_pdb: Path | None = None,
    ) -> None:
        """Load one visible structure, aligning it to the reference when provided."""
        pdb_path = pdb_path.expanduser().resolve()
        if not pdb_path.exists():
            raise FileNotFoundError(pdb_path)

        if reference_pdb is None:
            self._clear()
            self.cmd.load(str(pdb_path), "structure")
            view_selection = "structure"
        else:
            reference_pdb = reference_pdb.expanduser().resolve()
            if not reference_pdb.exists():
                raise FileNotFoundError(reference_pdb)
            self.cmd.delete("structure")
            prepared_name = self._prepared_structures.get(pdb_path)
            if self._reference_pdb == reference_pdb and prepared_name is not None:
                self.cmd.create("structure", prepared_name)
            else:
                self._load_reference(reference_pdb)
                self.cmd.load(str(pdb_path), "structure")
                self._align_to_reference("structure")
            view_selection = "reference"

        self.cmd.hide("everything", "all")
        self.cmd.dss("structure")
        self.cmd.show("cartoon", "structure")
        self._apply_render_settings()
        self._apply_residue_gradient("structure", self.sequence_id_names)
        if reference_pdb is not None and self._reference_view is not None:
            self.cmd.set_view(self._reference_view)
        else:
            self.cmd.orient(view_selection)
            self.cmd.zoom(view_selection, self.zoom_buffer, complete=1)
            if reference_pdb is not None:
                self._reference_view = self.cmd.get_view()

    def _clear(self) -> None:
        self.cmd.delete("all")
        self._reference_pdb = None
        self._reference_view = None
        self._prepared_structures = {}

    def _load_reference(self, reference_pdb: Path) -> None:
        if self._reference_pdb == reference_pdb:
            return
        self.cmd.delete("reference")
        self.cmd.load(str(reference_pdb), "reference")
        self._reference_pdb = reference_pdb
        self._reference_view = None
        self._prepared_structures = {}

    def _delete_prefixed(self, prefix: str) -> None:
        for name in self.cmd.get_names("objects"):
            if name.startswith(prefix):
                self.cmd.delete(name)

    def _align_to_reference(self, object_name: str) -> None:
        """Align an object to the loaded reference with the renderer's fixed policy."""
        mobile = f"{object_name} and ({ALIGN_SELECTION})"
        target = f"reference and ({ALIGN_SELECTION})"
        self.cmd.align(mobile, target, cycles=ALIGN_CYCLES)

    def _core_view_selection(self, object_names: Sequence[str]) -> str:
        """Prefer helices for camera bounds so long flexible tails do not dominate."""
        if not object_names:
            return "reference"

        selection = " or ".join(object_names)
        self.cmd.dss(selection)
        helix_selection = f"({selection}) and ss h"
        if self.cmd.count_atoms(helix_selection) > 0:
            return helix_selection
        return selection

    def save_png(self, output_png: Path) -> None:
        output_png = output_png.expanduser().resolve()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        self.cmd.png(
            str(output_png),
            width=self.width,
            height=self.height,
            dpi=self.dpi,
            ray=0,
        )

    def save_gif(
        self,
        output_gif: Path,
        *,
        frames: int,
        duration: float,
    ) -> None:
        """Save a slow rocking GIF using the current view as the center frame."""
        if frames < 1:
            raise ValueError("frames must be at least 1")

        import imageio.v2 as imageio

        output_gif = output_gif.expanduser().resolve()
        output_gif.parent.mkdir(parents=True, exist_ok=True)
        angles = self._rocking_angles(frames)

        with tempfile.TemporaryDirectory(prefix="pymol_frames_") as tmpdir:
            frame_paths = [Path(tmpdir) / f"frame_{i:03d}.png" for i in range(frames)]
            self._render_gif_frames(frame_paths, angles)
            imageio.mimsave(output_gif, [imageio.imread(path) for path in frame_paths], duration=duration, loop=0)

    def _define_colors(self) -> None:
        self.sequence_id_names = self._define_gradient("molstar_sequence_id", MOLSTAR_TURBO_STOPS)
        self.cmd.set_color("outline_gray", [0.08, 0.08, 0.08])

    def _define_gradient(self, prefix: str, stops: list[tuple[int, int, int]]) -> list[str]:
        names = []
        for i in range(PALETTE_SIZE):
            name = f"{prefix}_{i:03d}"
            self.cmd.set_color(name, list(interpolate_color(stops, i / (PALETTE_SIZE - 1))))
            names.append(name)
        return names

    def _apply_render_settings(self) -> None:
        """Apply the common cartoon style used by both PNG and GIF rendering."""
        self.cmd.bg_color("white")
        self.cmd.set("orthoscopic", 1)
        self.cmd.set("cartoon_fancy_helices", 1)
        self.cmd.set("cartoon_sampling", 20)
        self.cmd.set("antialias", 2)
        self.cmd.set("depth_cue", 0)
        self.cmd.set("ray_shadows", 0)
        self.cmd.set("specular", 0.10)
        self.cmd.set("ambient", 0.58)
        self.cmd.set("direct", 0.48)
        self.cmd.set("ray_opaque_background", 1)
        self.cmd.set("transparency_mode", 2)
        self.cmd.set("ray_trace_mode", 1)
        self.cmd.set("ray_trace_color", "outline_gray")
        self.cmd.set("ray_trace_gain", 0.12)

    def _apply_residue_gradient(self, selection: str, color_names: list[str]) -> None:
        """Color residues from N to C terminus with the supplied palette."""
        atoms = self.cmd.get_model(f"{selection} and name CA").atom
        if not atoms:
            atoms = self.cmd.get_model(selection).atom

        residues: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for atom in sorted(atoms, key=lambda item: item.index):
            key = (atom.segi, atom.chain, atom.resi)
            if key not in seen:
                seen.add(key)
                residues.append(key)

        if not residues:
            self.cmd.color(color_names[0], selection)
            return

        denom = max(len(residues) - 1, 1)
        for i, (segi, chain, resi) in enumerate(residues):
            color_name = color_names[round((i / denom) * (PALETTE_SIZE - 1))]
            self.cmd.color(color_name, self._residue_selection(selection, segi, chain, resi))

    @staticmethod
    def _residue_selection(selection: str, segi: str, chain: str, resi: str) -> str:
        parts = [selection]
        if segi:
            parts.append(f"segi {segi}")
        if chain:
            parts.append(f"chain {chain}")
        parts.append(f"resi {resi}")
        return " and ".join(parts)

    @staticmethod
    def _rocking_angles(frames: int, max_angle: float = ROCKING_ANGLE) -> list[float]:
        """Return absolute yaw angles: center -> right -> center -> left -> center."""
        if frames == 1:
            return [0.0]
        positions = [i / (frames - 1) for i in range(frames)]
        return [max_angle * math.sin(2 * math.pi * position) for position in positions]

    def _render_gif_frames(self, frame_paths: list[Path], angles: list[float]) -> None:
        base_view = self.cmd.get_view()
        for frame_path, angle in zip(frame_paths, angles, strict=True):
            # Reset every frame so rocking angles are absolute, not cumulative.
            self.cmd.set_view(base_view)
            self.cmd.turn("y", angle)
            self.cmd.png(
                str(frame_path),
                width=self.width,
                height=self.height,
                dpi=self.dpi,
                ray=0,
            )
