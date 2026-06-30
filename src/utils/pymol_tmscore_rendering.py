"""PyMOL renderer for per-residue TM-score colored prediction structures."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.utils.per_residue_tm_score import PerResidueTmScoreResult, ResidueTmScore
from src.utils.pymol_overlay_rendering import (
    GIF_DURATION_DEFAULT,
    GIF_DPI,
    GIF_FRAMES_DEFAULT,
    GIF_HEIGHT,
    GIF_WIDTH,
    PNG_DPI,
    PNG_HEIGHT,
    PNG_WIDTH,
    ROCKING_ANGLE,
    TITLE_COLOR,
    TITLE_FONT_CANDIDATES,
    ZOOM_BUFFER,
    add_silhouette_outline,
    add_title,
)
from src.utils.pymol_rendering import ALIGN_CYCLES, ALIGN_SELECTION, PALETTE_SIZE, interpolate_color

# Pale ice (score 1.0) -> yellow -> orange -> vivid red (score 0.5).
TM_SCORE_STOPS = [
    (0xF4, 0xF8, 0xFB),
    (0xFF, 0xF3, 0x6B),
    (0xFF, 0x9F, 0x1C),
    (0xE6, 0x25, 0x49),
]
TM_SCORE_CLIP_MIN = 0.5
TM_SCORE_CLIP_MAX = 1.0
TM_COLORBAR_WIDTH_FRAC = 0.022
TM_COLORBAR_HEIGHT_FRAC = 0.32
TM_COLORBAR_MARGIN_X_FRAC = 0.024
TM_VIEW_SHIFT_FRAC = 0.10


def clip_local_tm_score(score: float) -> float:
    """Clip local TM scores to the displayed range [0.5, 1.0]."""
    return max(TM_SCORE_CLIP_MIN, min(TM_SCORE_CLIP_MAX, score))


def local_tm_to_bfactor(score: float) -> float:
    """Map clipped local TM score to PyMOL B-factors on [0, 100].

    1.0 -> 0 (pale/good), 0.5 -> 100 (red/poor).
    """
    clipped = clip_local_tm_score(score)
    span = TM_SCORE_CLIP_MAX - TM_SCORE_CLIP_MIN
    return 100.0 * (TM_SCORE_CLIP_MAX - clipped) / span


def tm_score_palette_rgb(score: float) -> tuple[int, int, int]:
    """Return an RGB tuple for a local TM score using the display palette."""
    bfactor = local_tm_to_bfactor(score)
    t = bfactor / 100.0
    rgb = interpolate_color(TM_SCORE_STOPS, t)
    return tuple(int(channel * 255) for channel in rgb)


def _legend_font(image_width: int, *, scale: float = 1.0) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_size = max(34, int(image_width * 0.016 * scale))
    for font_path in TITLE_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def add_tm_score_colorbar(rgb: np.ndarray) -> np.ndarray:
    """Draw a TM-score colorbar for the clipped range 0.5 to 1.0."""
    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)
    font = _legend_font(img.width)
    tick_font = _legend_font(img.width, scale=0.95)

    bar_width = max(36, int(img.width * TM_COLORBAR_WIDTH_FRAC))
    bar_height = max(280, int(img.height * TM_COLORBAR_HEIGHT_FRAC))
    margin_x = max(36, int(img.width * TM_COLORBAR_MARGIN_X_FRAC))
    x0 = img.width - margin_x - bar_width
    y0 = (img.height - bar_height) // 2
    x1 = x0 + bar_width
    y1 = y0 + bar_height

    gradient = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)
    for row in range(bar_height):
        fraction = 1.0 - (row / (bar_height - 1) if bar_height > 1 else 0.0)
        score = TM_SCORE_CLIP_MIN + fraction * (TM_SCORE_CLIP_MAX - TM_SCORE_CLIP_MIN)
        gradient[row, :, :] = tm_score_palette_rgb(score)

    bar_img = Image.fromarray(gradient, mode="RGB")
    img.paste(bar_img, (x0, y0))
    draw.rectangle([x0, y0, x1, y1], outline=TITLE_COLOR, width=3)

    label_x = x0 - max(18, int(img.width * 0.014))
    draw.text((label_x, y0), f"{TM_SCORE_CLIP_MAX:.1f}", fill=TITLE_COLOR, font=tick_font, anchor="ra")
    draw.text((label_x, y1), f"{TM_SCORE_CLIP_MIN:.1f}", fill=TITLE_COLOR, font=tick_font, anchor="ra")
    draw.text(
        (x0 + bar_width // 2, y1 + max(16, int(img.height * 0.014))),
        "TM",
        fill=TITLE_COLOR,
        font=font,
        anchor="mt",
    )
    return np.array(img)


class TmScoreRenderer:
    """Render aligned predictions colored by per-residue local TM score."""

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
        self._tm_score_color_names = self._define_tm_score_colors()

    def __enter__(self) -> "TmScoreRenderer":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self.cmd.quit()

    def prepare_fixed_view(self, reference_pdb: Path) -> None:
        reference_pdb = reference_pdb.expanduser().resolve()
        if not reference_pdb.exists():
            raise FileNotFoundError(reference_pdb)

        self._load_reference(reference_pdb)
        view_selection = self._core_view_selection("ref")
        self.cmd.orient("ref")
        self.cmd.zoom(view_selection, self.zoom_buffer, complete=1)
        self._reference_view = self.cmd.get_view()
        self.cmd.hide("everything", "all")

    def load_tmscore_structure(
        self,
        prediction_pdb: Path,
        reference_pdb: Path,
        scores: PerResidueTmScoreResult,
    ) -> None:
        """Load prediction, align to reference, and color by residue TM score."""
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
        self.cmd.dss("pred")
        self.cmd.align(f"pred and ({ALIGN_SELECTION})", f"ref and ({ALIGN_SELECTION})", cycles=ALIGN_CYCLES)

        self.cmd.hide("everything", "all")
        self.cmd.show("cartoon", "pred")
        self._apply_cartoon_style()
        self._apply_tm_score_colors(scores.residues)

        if self._reference_view is not None:
            self.cmd.set_view(self._reference_view)
        self._shift_content_left("pred")

    def save_tmscore_png(self, output_png: Path, *, title: str | None = None) -> None:
        output_png = output_png.expanduser().resolve()
        output_png.parent.mkdir(parents=True, exist_ok=True)

        self._apply_png_render_settings()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            alpha_path = Path(tmp.name)

        try:
            self.cmd.ray(self.png_width, self.png_height)
            self.cmd.png(str(alpha_path), dpi=self.png_dpi)
            rgba = np.array(Image.open(alpha_path).convert("RGBA"))
            rgb = self._finish_png_frame(rgba, title=title)
            Image.fromarray(rgb, mode="RGB").save(output_png, dpi=(self.png_dpi, self.png_dpi))
        finally:
            alpha_path.unlink(missing_ok=True)

    def save_tmscore_gif(
        self,
        output_gif: Path,
        *,
        frames: int = GIF_FRAMES_DEFAULT,
        duration: float = GIF_DURATION_DEFAULT,
    ) -> None:
        """Save a rocking GIF with opaque background, colorbar, and no silhouette."""
        if frames < 1:
            raise ValueError("frames must be at least 1")

        import imageio.v2 as imageio

        output_gif = output_gif.expanduser().resolve()
        output_gif.parent.mkdir(parents=True, exist_ok=True)
        angles = self._rocking_angles(frames)

        self._apply_gif_render_settings()
        with tempfile.TemporaryDirectory(prefix="tmscore_gif_") as tmpdir:
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
                [add_tm_score_colorbar(np.array(Image.open(path).convert("RGB"))) for path in frame_paths],
                duration=duration,
                loop=0,
            )

    def _finish_png_frame(self, rgba: np.ndarray, *, title: str | None) -> np.ndarray:
        rgb = add_tm_score_colorbar(add_silhouette_outline(rgba))
        if title:
            rgb = add_title(rgb, title)
        return rgb

    def _define_tm_score_colors(self) -> list[str]:
        names: list[str] = []
        for index in range(PALETTE_SIZE):
            name = f"tmscore_{index:03d}"
            rgb = interpolate_color(TM_SCORE_STOPS, index / (PALETTE_SIZE - 1))
            self.cmd.set_color(name, list(rgb))
            names.append(name)
        return names

    def _apply_tm_score_colors(self, residues: tuple[ResidueTmScore, ...]) -> None:
        for entry in residues:
            bfactor = local_tm_to_bfactor(entry.score)
            selection = f"pred and chain {entry.chain_id} and resi {entry.resi}"
            self.cmd.alter(selection, f"b={bfactor:.6f}")

        self.cmd.spectrum(
            "b",
            " ".join(self._tm_score_color_names),
            "pred",
            minimum=0,
            maximum=100,
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

    def _shift_content_left(self, selection: str, fraction: float = TM_VIEW_SHIFT_FRAC) -> None:
        """Translate structure left on screen to leave room for the colorbar."""
        extent = self.cmd.get_extent(selection)
        if extent is None:
            return

        width = max(extent[1][axis] - extent[0][axis] for axis in range(3))
        shift = width * fraction
        view = self.cmd.get_view()
        horizontal = np.array([view[0], view[3], view[6]], dtype=float)
        norm = np.linalg.norm(horizontal)
        if norm < 1e-8:
            return

        delta = (-shift * horizontal / norm).tolist()
        self.cmd.translate(delta, selection)

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

    def _apply_png_render_settings(self) -> None:
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
        self.cmd.set("antialias", 4)
        self.cmd.set("hash_max", 300)
        self.cmd.set("ray_opaque_background", 0)

    def _apply_gif_render_settings(self) -> None:
        self._apply_png_render_settings()
        self.cmd.set("antialias", 2)
        self.cmd.set("ray_opaque_background", 1)

    @staticmethod
    def _rocking_angles(frames: int, max_angle: float = ROCKING_ANGLE) -> list[float]:
        if frames == 1:
            return [0.0]

        half = frames // 2
        angles: list[float] = []
        for i in range(half):
            t = i / (half - 1) if half > 1 else 0.0
            angles.append(-max_angle * math.sin(math.pi * t))
        remainder = frames - half
        for i in range(remainder):
            t = i / (remainder - 1) if remainder > 1 else 0.0
            angles.append(max_angle * math.sin(math.pi * t))
        return angles
