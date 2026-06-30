"""PyMOL renderer for reference vs prediction overlay figures.

Renders colored reference (teal/amber by state) + gray prediction with
cylindrical helices, optional silhouette outlines on PNGs, and rocking GIFs.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.utils.pymol_rendering import ALIGN_CYCLES, ALIGN_SELECTION

OVERLAY_TEAL = [0.455, 0.910, 0.753]  # #74e8c0 — state_1 reference
OVERLAY_AMBER = [1.0, 0.769, 0.012]  # #ffc403 — state_2 reference
MASK_HIGHLIGHT = [0.95, 0.15, 0.15]  # red — query-masked residues on prediction

REFERENCE_COLORS = {
    "state_1": OVERLAY_TEAL,
    "state_2": OVERLAY_AMBER,
}
OUTLINE_COLOR = 40.0
SILHOUETTE_KERNEL = 7
SILHOUETTE_BLUR = 1.0
SILHOUETTE_STRENGTH = 0.85

PNG_WIDTH = 4000
PNG_HEIGHT = 3200
PNG_DPI = 300

GIF_WIDTH = 800
GIF_HEIGHT = 640
GIF_DPI = 150
GIF_FRAMES_DEFAULT = 32
GIF_DURATION_DEFAULT = 0.08
ROCKING_ANGLE = 20.0

ZOOM_BUFFER = 5.0

TITLE_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
)
TITLE_COLOR = (40, 40, 40)
TITLE_TOP_MARGIN_FRAC = 0.025
TITLE_FONT_SIZE_FRAC = 0.028


def add_silhouette_outline(
    rgba: np.ndarray,
    *,
    kernel: int = SILHOUETTE_KERNEL,
    blur: float = SILHOUETTE_BLUR,
    strength: float = SILHOUETTE_STRENGTH,
) -> np.ndarray:
    """Composite RGBA over white and blend a dark outline at the silhouette edge."""
    alpha = rgba[:, :, 3]
    mask = (alpha > 10).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, mode="L")
    dilated = np.array(mask_img.filter(ImageFilter.MaxFilter(kernel)), dtype=np.int16)
    eroded = np.array(mask_img.filter(ImageFilter.MinFilter(kernel)), dtype=np.int16)
    edge = np.clip(dilated - eroded, 0, 255).astype(np.uint8)
    edge_f = (
        np.array(Image.fromarray(edge, mode="L").filter(ImageFilter.GaussianBlur(radius=blur)), dtype=np.float32)
        / 255.0
    )

    alpha_f = rgba[:, :, 3:4].astype(np.float32) / 255.0
    result = rgba[:, :, :3].astype(np.float32) * alpha_f + 255.0 * (1.0 - alpha_f)
    for channel in range(3):
        result[:, :, channel] = (
            result[:, :, channel] * (1.0 - edge_f * strength) + OUTLINE_COLOR * edge_f * strength
        )
    return np.clip(result, 0, 255).astype(np.uint8)


def _title_font(image_width: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_size = max(48, int(image_width * TITLE_FONT_SIZE_FRAC))
    for font_path in TITLE_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def add_title(rgb: np.ndarray, title: str) -> np.ndarray:
    """Draw a centered experiment title at the top of an overlay PNG."""
    if not title:
        return rgb

    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)
    font = _title_font(img.width)
    bbox = draw.textbbox((0, 0), title, font=font)
    text_width = bbox[2] - bbox[0]
    x = (img.width - text_width) // 2
    y = int(img.height * TITLE_TOP_MARGIN_FRAC)
    draw.text((x, y), title, fill=TITLE_COLOR, font=font)
    return np.array(img)


class OverlayRenderer:
    """Reusable PyMOL session for reference/prediction overlay rendering."""

    def __init__(
        self,
        *,
        png_width: int = PNG_WIDTH,
        png_height: int = PNG_HEIGHT,
        png_dpi: int = PNG_DPI,
        gif_width: int = GIF_WIDTH,
        gif_height: int = GIF_HEIGHT,
        gif_dpi: int = GIF_DPI,
        zoom_buffer: float = ZOOM_BUFFER,
    ):
        import pymol
        from pymol import cmd

        self.cmd = cmd
        self.png_width = png_width
        self.png_height = png_height
        self.png_dpi = png_dpi
        self.gif_width = gif_width
        self.gif_height = gif_height
        self.gif_dpi = gif_dpi
        self.zoom_buffer = zoom_buffer
        self._reference_pdb: Path | None = None
        self._reference_view = None

        pymol.finish_launching(["pymol", "-cq"])
        for name, rgb in (
            ("overlay_teal", OVERLAY_TEAL),
            ("overlay_amber", OVERLAY_AMBER),
            ("overlay_mask", MASK_HIGHLIGHT),
        ):
            self.cmd.set_color(name, rgb)

    def __enter__(self) -> "OverlayRenderer":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self.cmd.quit()

    def prepare_fixed_view(self, reference_pdb: Path) -> None:
        """Prepare one stable camera from the reference structure."""
        reference_pdb = reference_pdb.expanduser().resolve()
        if not reference_pdb.exists():
            raise FileNotFoundError(reference_pdb)

        self._load_reference(reference_pdb)
        view_selection = self._core_view_selection("ref")
        self.cmd.orient("ref")
        self.cmd.zoom(view_selection, self.zoom_buffer, complete=1)
        self._reference_view = self.cmd.get_view()
        self.cmd.hide("everything", "all")

    def load_overlay(
        self,
        prediction_pdb: Path,
        reference_pdb: Path,
        *,
        state_key: str = "state_1",
        highlight_residues: list[int] | None = None,
    ) -> None:
        """Load reference and prediction, align prediction onto reference."""
        prediction_pdb = prediction_pdb.expanduser().resolve()
        reference_pdb = reference_pdb.expanduser().resolve()
        if not prediction_pdb.exists():
            raise FileNotFoundError(prediction_pdb)
        if not reference_pdb.exists():
            raise FileNotFoundError(reference_pdb)

        self._load_reference(reference_pdb)
        self.cmd.delete("pred")
        self.cmd.load(str(prediction_pdb), "pred")
        self.cmd.remove("solvent")
        self.cmd.dss("ref")
        self.cmd.dss("pred")
        self.cmd.align(f"pred and ({ALIGN_SELECTION})", f"ref and ({ALIGN_SELECTION})", cycles=ALIGN_CYCLES)

        self.cmd.hide("everything", "all")
        self.cmd.show("cartoon", "ref")
        self.cmd.show("cartoon", "pred")
        self._apply_cartoon_style()
        ref_color = "overlay_teal" if state_key == "state_1" else "overlay_amber"
        self.cmd.color(ref_color, "ref")
        self.cmd.color("gray85", "pred")
        self._highlight_query_mask_residues(highlight_residues)

        if self._reference_view is not None:
            self.cmd.set_view(self._reference_view)

    def _highlight_query_mask_residues(self, highlight_residues: list[int] | None) -> None:
        if not highlight_residues:
            return

        residue_ids = "+".join(str(residue) for residue in highlight_residues)
        self.cmd.color("overlay_mask", f"pred and resi {residue_ids}")

    def save_overlay_png(self, output_png: Path, *, title: str | None = None) -> None:
        """Ray-trace with transparent background and apply silhouette outline."""
        output_png = output_png.expanduser().resolve()
        output_png.parent.mkdir(parents=True, exist_ok=True)

        self._apply_png_render_settings()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            alpha_path = Path(tmp.name)

        try:
            self.cmd.ray(self.png_width, self.png_height)
            self.cmd.png(str(alpha_path), dpi=self.png_dpi)
            rgba = np.array(Image.open(alpha_path).convert("RGBA"))
            rgb = add_silhouette_outline(rgba)
            if title:
                rgb = add_title(rgb, title)
            Image.fromarray(rgb, mode="RGB").save(output_png, dpi=(self.png_dpi, self.png_dpi))
        finally:
            alpha_path.unlink(missing_ok=True)

    def save_overlay_gif(
        self,
        output_gif: Path,
        *,
        frames: int = GIF_FRAMES_DEFAULT,
        duration: float = GIF_DURATION_DEFAULT,
    ) -> None:
        """Save a rocking GIF with opaque background and no silhouette."""
        if frames < 1:
            raise ValueError("frames must be at least 1")

        import imageio.v2 as imageio

        output_gif = output_gif.expanduser().resolve()
        output_gif.parent.mkdir(parents=True, exist_ok=True)
        angles = self._rocking_angles(frames)

        self._apply_gif_render_settings()
        with tempfile.TemporaryDirectory(prefix="overlay_gif_") as tmpdir:
            frame_paths = [Path(tmpdir) / f"frame_{i:03d}.png" for i in range(frames)]
            base_view = self.cmd.get_view()
            for frame_path, angle in zip(frame_paths, angles, strict=True):
                self.cmd.set_view(base_view)
                self.cmd.turn("y", angle)
                self.cmd.ray(self.gif_width, self.gif_height)
                self.cmd.png(
                    str(frame_path),
                    width=self.gif_width,
                    height=self.gif_height,
                    dpi=self.gif_dpi,
                )
            imageio.mimsave(
                output_gif,
                [imageio.imread(path) for path in frame_paths],
                duration=duration,
                loop=0,
            )

    def _load_reference(self, reference_pdb: Path) -> None:
        if self._reference_pdb == reference_pdb:
            return
        self.cmd.delete("all")
        self.cmd.load(str(reference_pdb), "ref")
        self._reference_pdb = reference_pdb
        self._reference_view = None

    def _core_view_selection(self, selection: str) -> str:
        self.cmd.dss(selection)
        helix_selection = f"({selection}) and ss h"
        if self.cmd.count_atoms(helix_selection) > 0:
            return helix_selection
        return selection

    def _apply_cartoon_style(self) -> None:
        self.cmd.set("cartoon_cylindrical_helices", 1)
        self.cmd.set("cartoon_fancy_helices", 1)
        self.cmd.set("cartoon_sampling", 40)
        self.cmd.set("cartoon_loop_quality", 20)
        self.cmd.set("cartoon_tube_quality", 20)
        self.cmd.set("cartoon_putty_quality", 20)
        self.cmd.set("cartoon_loop_radius", 0.15)
        self.cmd.set("cartoon_tube_radius", 0.22)
        self.cmd.set("cartoon_oval_length", 1.0)
        self.cmd.set("cartoon_oval_width", 0.25)

    def _apply_lighting(self) -> None:
        self.cmd.bg_color("white")
        self.cmd.set("orthoscopic", 1)
        self.cmd.set("ambient", 0.28)
        self.cmd.set("direct", 0.65)
        self.cmd.set("specular", 0.22)
        self.cmd.set("shininess", 25)
        self.cmd.set("reflect", 0.5)
        self.cmd.set("depth_cue", 1)
        self.cmd.set("fog_start", 0.45)
        self.cmd.set("ray_shadows", 1)
        self.cmd.set("ray_shadow_decay_factor", 0.04)
        self.cmd.set("ray_trace_mode", 0)

    def _apply_png_render_settings(self) -> None:
        self._apply_lighting()
        self.cmd.set("antialias", 4)
        self.cmd.set("hash_max", 300)
        self.cmd.set("ray_opaque_background", 0)

    def _apply_gif_render_settings(self) -> None:
        self._apply_lighting()
        self.cmd.set("antialias", 2)
        self.cmd.set("ray_opaque_background", 1)

    @staticmethod
    def _rocking_angles(frames: int, max_angle: float = ROCKING_ANGLE) -> list[float]:
        """Return yaw angles: first half swings left and back, second half swings right."""
        if frames == 1:
            return [0.0]

        half = frames // 2
        angles: list[float] = []

        # Left: center -> left -> center
        for i in range(half):
            t = i / (half - 1) if half > 1 else 0.0
            angles.append(-max_angle * math.sin(math.pi * t))

        # Right: center -> right -> center
        remainder = frames - half
        for i in range(remainder):
            t = i / (remainder - 1) if remainder > 1 else 0.0
            angles.append(max_angle * math.sin(math.pi * t))

        return angles
