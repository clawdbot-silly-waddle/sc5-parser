"""SC v5 file parser.

Decodes the binary SC v5 container (header → FileDescriptor FlatBuffer →
ZSTD-compressed inner stream) and exposes shapes, movie-clips, textures
and named exports for downstream extraction.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import zstandard
from PIL import Image

from sc5_parser._schemas.sc.flash.SC2.CompressedMovieClips import CompressedMovieClips
from sc5_parser._schemas.sc.flash.SC2.DataStorage import DataStorage
from sc5_parser._schemas.sc.flash.SC2.ExportNames import ExportNames
from sc5_parser._schemas.sc.flash.SC2.ExternalMatrixBanks import ExternalMatrixBanks
from sc5_parser._schemas.sc.flash.SC2.FileDescriptor import FileDescriptor
from sc5_parser._schemas.sc.flash.SC2.MovieClipModifiers import MovieClipModifiers
from sc5_parser._schemas.sc.flash.SC2.MovieClips import MovieClips
from sc5_parser._schemas.sc.flash.SC2.Precision import Precision
from sc5_parser._schemas.sc.flash.SC2.Shapes import Shapes
from sc5_parser._schemas.sc.flash.SC2.TextFields import TextFields
from sc5_parser._schemas.sc.flash.SC2.Textures import Textures
from sc5_parser.renderer import render_command

ShapeDict = dict[str, Any]

# Sentinel: "no matrix" / "no color transform" in frame elements
_NO_TRANSFORM = 0xFFFF

# MovieClipModifier types
_MOD_MASK = 38      # Defines the start of a mask group; next child is the mask shape
_MOD_MASKED = 39    # Children after this are clipped by the mask
_MOD_UNMASKED = 40  # End of masked group; children render normally


def _precision_divisor(precision: int) -> float:
    """Return the divisor for a Precision enum value.

    Matches SupercellSWF2CompileTable::get_precision_multiplier in the
    C++ reference — values were multiplied by this factor during encoding
    so we divide to recover the original float.
    """
    if precision == Precision.Twip:
        return 20.0
    if precision == Precision.Optimized:
        return 1024.0
    # None_ (0) and Default (1) both use 1.0
    return 1.0


@dataclass
class Matrix2x3:
    """Affine 2×3 transformation matrix (a b c d tx ty)."""
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    tx: float = 0.0
    ty: float = 0.0

    IDENTITY: Matrix2x3 = None  # type: ignore[assignment]  # set below

    def __matmul__(self, other: Matrix2x3) -> Matrix2x3:
        """Compose two 2×3 affine matrices: ``self @ other``."""
        return Matrix2x3(
            a=self.a * other.a + self.b * other.c,
            b=self.a * other.b + self.b * other.d,
            c=self.c * other.a + self.d * other.c,
            d=self.c * other.b + self.d * other.d,
            tx=self.a * other.tx + self.b * other.ty + self.tx,
            ty=self.c * other.tx + self.d * other.ty + self.ty,
        )


Matrix2x3.IDENTITY = Matrix2x3()


@dataclass
class ColorTransform:
    """RGBA color transform: new = old * mul/255 + add."""
    r_mul: int = 255
    g_mul: int = 255
    b_mul: int = 255
    alpha: int = 255
    r_add: int = 0
    g_add: int = 0
    b_add: int = 0

    IDENTITY: ColorTransform = None  # type: ignore[assignment]

    def apply(self, img: Image.Image) -> Image.Image:
        """Apply this color transform to an RGBA image."""
        if self is ColorTransform.IDENTITY:
            return img
        if (self.r_mul == 255 and self.g_mul == 255 and self.b_mul == 255
                and self.alpha == 255
                and self.r_add == 0 and self.g_add == 0 and self.b_add == 0):
            return img
        arr = np.array(img, dtype=np.int32)
        r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
        arr[:, :, 0] = np.clip(r * self.r_mul // 255 + self.r_add, 0, 255)
        arr[:, :, 1] = np.clip(g * self.g_mul // 255 + self.g_add, 0, 255)
        arr[:, :, 2] = np.clip(b * self.b_mul // 255 + self.b_add, 0, 255)
        arr[:, :, 3] = np.clip(a * self.alpha // 255, 0, 255)
        return Image.fromarray(arr.astype(np.uint8), "RGBA")


ColorTransform.IDENTITY = ColorTransform()


@dataclass
class FrameElement:
    """One visible child in a MovieClip frame."""
    child_index: int
    matrix_index: int
    color_index: int


@dataclass
class TextFieldData:
    """Parsed TextField display object."""
    id: int
    font_name: str = ""
    text: str = ""
    typography_file: str = ""
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0
    font_color: int = 0          # RGBA packed uint32
    outline_color: int = 0       # RGBA packed uint32
    font_size: int = 0
    align: int = 0
    styles: int = 0              # bit flags: bold, italic, multiline, etc.


@dataclass
class ScalingGrid:
    """9-patch / 9-slice rectangle for a MovieClip."""
    left: float = 0.0
    top: float = 0.0
    right: float = 0.0
    bottom: float = 0.0


@dataclass
class MovieClipData:
    """Parsed MovieClip with frame element data."""
    id: int
    children_ids: list[int] = field(default_factory=list)
    children_names: list[str] = field(default_factory=list)
    children_blending: list[int] = field(default_factory=list)
    frame_elements_offset: int = 0xFFFFFFFF
    matrix_bank_index: int = 0
    frame_element_counts: list[int] = field(default_factory=list)
    frame_labels: list[str] = field(default_factory=list)
    framerate: int = 24  # FPS; 0 means "use default (24)"
    scaling_grid: ScalingGrid | None = None


@dataclass
class RenderContext:
    """Optional rendering configuration passed through recursive render calls.

    Allows consumers to customize frame selection, child filtering, and
    content injection without modifying the SC5File instance.
    """
    frame_finder: Callable[[int, str], int] | None = None
    """Custom frame resolver.  Called as ``frame_finder(mc_id, label)``
    and should return a frame index.  When ``None`` the default
    :meth:`SC5File.find_frame_by_label` is used."""

    render_children: dict[int, frozenset[int]] | None = None
    """Per-MC child filter.  Keys are MC IDs; values are frozensets of
    allowed ``child_index`` values.  When a MC ID is present in this
    dict only the listed children are rendered."""

    inject_in_mask: dict[int, list[tuple[Image.Image, float, float, int]]] | None = None
    """Content to inject into mask groups.  Keys are MC IDs; values are
    lists of ``(image, x, y, blend_mode)`` tuples that get composited
    into the mask group at the MASKED modifier boundary."""

    overlay_after_child: dict[int, list[tuple[Image.Image, float, float, int]]] | None = None
    """Extra parts to insert after a specific child index during top-level
    MC rendering.  Keys are child indices; values are lists of
    ``(image, x, y, blend_mode)`` tuples.  Only consumed at depth 0 of
    the root ``render_object`` call (same scope as *child_labels*)."""


class SC5File:
    """Parsed representation of an SC v5 file."""

    def __init__(self, sc_path: str | Path) -> None:
        self.sc_path = Path(sc_path)
        self.shapes: list[ShapeDict] = []
        self.exports: dict[str, int] = {}  # name → movie-clip / shape id
        self.textures: list[dict[str, Any]] = []
        self.movie_clip_data: dict[int, MovieClipData] = {}
        self.text_fields: dict[int, TextFieldData] = {}  # id → text field
        self.modifiers: dict[int, int] = {}  # id → modifier type (38/39/40)
        self.strings: list[str] = []
        self.vertices: list[tuple[float, float, int, int]] = []
        self._scaling_rects: list[ScalingGrid] = []  # from DataStorage.rectangles
        self._shape_id_to_idx: dict[int, list[int]] = {}
        self._frame_elements: np.ndarray | None = None
        self._matrix_banks: list[list[Matrix2x3]] = []
        self._color_banks: list[list[ColorTransform]] = []
        self._parse()

    # ------------------------------------------------------------------
    def _parse(self) -> None:
        raw = self.sc_path.read_bytes()

        if len(raw) < 10:
            raise ValueError(f"Invalid SC file: too small ({len(raw)} bytes)")
        if raw[:2] != b"SC":
            raise ValueError(f"Not an SC file: {self.sc_path}")
        version = struct.unpack("<I", raw[2:6])[0]
        if version != 5:
            raise ValueError(f"Unsupported SC version {version} (expected 5)")

        fd_size = struct.unpack("<I", raw[6:10])[0]
        fd_buf = bytes(raw[10 : 10 + fd_size])
        fd = FileDescriptor.GetRootAs(fd_buf, 0)

        # Precision divisors for half-precision matrix decoding
        scale_div = _precision_divisor(fd.ScalePrecision())
        trans_div = _precision_divisor(fd.TranslationPrecision())

        comp_start = 10 + fd_size
        comp_data = (
            raw[comp_start : comp_start + fd.CompressedSize()]
            if fd.CompressedSize()
            else raw[comp_start:]
        )

        inner = zstandard.ZstdDecompressor().decompress(
            comp_data, max_output_size=100 * 1024 * 1024
        )

        # External matrix bank data sits after the compressed inner stream.
        ext_mb_size = fd.ExternalMatrixBankSize() or 0
        ext_mb_data: bytes | None = None
        if ext_mb_size > 0:
            ext_start = comp_start + (fd.CompressedSize() or 0)
            ext_mb_data = bytes(raw[ext_start : ext_start + ext_mb_size])

        # --- DataStorage --------------------------------------------------
        ds_size = struct.unpack("<I", inner[0:4])[0]
        ds = DataStorage.GetRootAs(bytes(inner[4 : 4 + ds_size]), 0)

        self.strings = [
            (ds.Strings(i).decode() if ds.Strings(i) else "")
            for i in range(ds.StringsLength())
        ]

        bp_len = ds.ShapesBitmapPoinsLength()
        if bp_len:
            bp_data = bytes(ds.ShapesBitmapPoins(i) for i in range(bp_len))
            for i in range(bp_len // 12):
                off = i * 12
                x = struct.unpack("<f", bp_data[off : off + 4])[0]
                y = struct.unpack("<f", bp_data[off + 4 : off + 8])[0]
                u = struct.unpack("<H", bp_data[off + 8 : off + 10])[0]
                v = struct.unpack("<H", bp_data[off + 10 : off + 12])[0]
                self.vertices.append((x, y, u, v))

        # --- Frame elements (u16 array) -----------------------------------
        # Generated FlatBuffer code has a bug: it accesses [ushort] as [ubyte].
        # We read the raw vector data correctly as u16 here.
        tab = ds._tab
        fe_field_off = tab.Offset(12)  # vtable slot for movieclips_frame_elements
        if fe_field_off:
            fe_vec_off = tab.Vector(fe_field_off)
            fe_count = struct.unpack("<I", tab.Bytes[fe_vec_off - 4 : fe_vec_off])[0]
            self._frame_elements = np.frombuffer(
                tab.Bytes, dtype="<u2", offset=fe_vec_off, count=fe_count
            )
        else:
            self._frame_elements = np.array([], dtype="<u2")

        # --- Scaling grid rectangles from DataStorage ----------------------
        for ri in range(ds.RectanglesLength()):
            r = ds.Rectangles(ri)
            self._scaling_rects.append(
                ScalingGrid(left=r.Left(), top=r.Top(), right=r.Right(), bottom=r.Bottom())
            )

        # --- Matrix banks --------------------------------------------------
        for bi in range(ds.MatrixBanksLength()):
            bank_fb = ds.MatrixBanks(bi)
            matrices: list[Matrix2x3] = []
            # Prefer full-precision matrices; fall back to half-precision
            if bank_fb.MatricesLength() > 0:
                for mi in range(bank_fb.MatricesLength()):
                    m = bank_fb.Matrices(mi)
                    matrices.append(
                        Matrix2x3(a=m.A(), b=m.B(), c=m.C(), d=m.D(), tx=m.Tx(), ty=m.Ty())
                    )
            elif bank_fb.HalfMatricesLength() > 0:
                for mi in range(bank_fb.HalfMatricesLength()):
                    m = bank_fb.HalfMatrices(mi)
                    matrices.append(Matrix2x3(
                        a=m.A() / scale_div,
                        b=m.B() / scale_div,
                        c=m.C() / scale_div,
                        d=m.D() / scale_div,
                        tx=m.Tx() / trans_div,
                        ty=m.Ty() / trans_div,
                    ))
            self._matrix_banks.append(matrices)
            colors: list[ColorTransform] = []
            for ci in range(bank_fb.ColorsLength()):
                c = bank_fb.Colors(ci)
                colors.append(ColorTransform(
                    r_mul=c.RMul(), g_mul=c.GMul(), b_mul=c.BMul(),
                    alpha=c.Alpha(),
                    r_add=c.RAdd(), g_add=c.GAdd(), b_add=c.BAdd(),
                ))
            self._color_banks.append(colors)

        # --- External matrix banks (appended after the internal banks) -----
        if ext_mb_data is not None and len(ext_mb_data) >= 4:
            try:
                desc_size = struct.unpack_from("<I", ext_mb_data, 0)[0]
                embs = ExternalMatrixBanks.GetRootAs(
                    bytes(ext_mb_data[4 : 4 + desc_size]), 0
                )
                banks_data_offset = 4 + desc_size
                for bi in range(embs.BanksLength()):
                    eb = embs.Banks(bi)
                    coff = banks_data_offset + eb.CompressedDataOffset()
                    csz = eb.CompressedDataSize()
                    dsz = eb.DecompressedDataSize()
                    if csz == 0 or dsz == 0:
                        continue
                    bank_data = zstandard.ZstdDecompressor().decompress(
                        ext_mb_data[coff : coff + csz],
                        max_output_size=dsz,
                    )
                    ext_matrices: list[Matrix2x3] = []
                    off = 0
                    # Float matrices (24 bytes each)
                    for _ in range(eb.FloatMatrixCount()):
                        a, b, c, d, tx, ty = struct.unpack_from("<6f", bank_data, off)
                        ext_matrices.append(Matrix2x3(a=a, b=b, c=c, d=d, tx=tx, ty=ty))
                        off += 24
                    # Skip compressed matrix data (complex RLE codec, not decoded here)
                    off += eb.CompressedMatrixDataSize() * 4
                    # Short matrices (12 bytes each, hardcoded /1024 and /20 divisors)
                    for _ in range(eb.ShortMatrixCount()):
                        sa, sb, sc_, sd, stx, sty = struct.unpack_from("<6h", bank_data, off)
                        ext_matrices.append(Matrix2x3(
                            a=sa / 1024.0, b=sb / 1024.0,
                            c=sc_ / 1024.0, d=sd / 1024.0,
                            tx=stx / 20.0, ty=sty / 20.0,
                        ))
                        off += 12
                    self._matrix_banks.append(ext_matrices)
                    # Color transforms (7 bytes each: r_mul, g_mul, b_mul, alpha, r_add, g_add, b_add)
                    ct_off = (eb.FloatMatrixCount() * 24
                              + eb.CompressedMatrixDataSize() * 4
                              + eb.ShortMatrixDataSize() * 2)
                    ext_colors: list[ColorTransform] = []
                    for _ in range(eb.ColorTransformCount()):
                        vals = struct.unpack_from("<7B", bank_data, ct_off)
                        ext_colors.append(ColorTransform(
                            r_mul=vals[0], g_mul=vals[1], b_mul=vals[2],
                            alpha=vals[3],
                            r_add=vals[4], g_add=vals[5], b_add=vals[6],
                        ))
                        ct_off += 7
                    self._color_banks.append(ext_colors)
            except Exception:
                pass  # Graceful degradation if external bank parsing fails

        # --- Chunked resources at resources_offset ------------------------
        pos = fd.ResourcesOffset()

        # ExportNames
        en_size = struct.unpack("<I", inner[pos : pos + 4])[0]
        en = ExportNames.GetRootAs(bytes(inner[pos + 4 : pos + 4 + en_size]), 0)
        for i in range(en.ObjectIdsLength()):
            oid = en.ObjectIds(i)
            nref = en.NameRefIds(i) if i < en.NameRefIdsLength() else 0
            name = (
                self.strings[nref - 1]
                if 0 < nref <= len(self.strings)
                else ""
            )
            if name:
                self.exports[name] = oid
        pos += 4 + en_size

        # TextFields
        tf_size = struct.unpack("<I", inner[pos : pos + 4])[0]
        if tf_size > 0:
            tf = TextFields.GetRootAs(bytes(inner[pos + 4 : pos + 4 + tf_size]), 0)
            for i in range(tf.TextfieldsLength()):
                tfd = tf.Textfields(i)
                tf_id = tfd.Id()
                font_ref = tfd.FontNameRefId()
                text_ref = tfd.TextRefId()
                typo_ref = tfd.TypographyRefId()
                self.text_fields[tf_id] = TextFieldData(
                    id=tf_id,
                    font_name=(
                        self.strings[font_ref - 1]
                        if 0 < font_ref <= len(self.strings) else ""
                    ),
                    text=(
                        self.strings[text_ref - 1]
                        if 0 < text_ref <= len(self.strings) else ""
                    ),
                    typography_file=(
                        self.strings[typo_ref - 1]
                        if 0 < typo_ref <= len(self.strings) else ""
                    ),
                    left=tfd.Left(),
                    top=tfd.Top(),
                    right=tfd.Right(),
                    bottom=tfd.Bottom(),
                    font_color=tfd.FontColor(),
                    outline_color=tfd.OutlineColor(),
                    font_size=tfd.FontSize(),
                    align=tfd.Align(),
                    styles=tfd.Styles(),
                )
        pos += 4 + tf_size

        # Shapes
        sh_size = struct.unpack("<I", inner[pos : pos + 4])[0]
        sh = Shapes.GetRootAs(bytes(inner[pos + 4 : pos + 4 + sh_size]), 0)
        for i in range(sh.ShapesLength()):
            shape = sh.Shapes(i)
            commands: list[dict[str, Any]] = []
            for j in range(shape.CommandsLength()):
                cmd = shape.Commands(j)
                verts = [
                    self.vertices[cmd.PointsOffset() + k]
                    for k in range(cmd.PointsCount())
                ]
                commands.append(
                    {"texture_index": cmd.TextureIndex(), "vertices": verts}
                )
            sid = shape.Id()
            self.shapes.append({"id": sid, "commands": commands})
            self._shape_id_to_idx.setdefault(sid, []).append(
                len(self.shapes) - 1
            )
        pos += 4 + sh_size

        # MovieClips (regular or compressed variant)
        mc_size = struct.unpack("<I", inner[pos : pos + 4])[0]
        mc_buf = bytes(inner[pos + 4 : pos + 4 + mc_size])
        mc = MovieClips.GetRootAs(mc_buf, 0)

        if mc.MovieclipsLength() > 0:
            self._parse_movie_clips(mc)
        else:
            # Try CompressedMovieClips variant
            try:
                cmc = CompressedMovieClips.GetRootAs(mc_buf, 0)
                if cmc.MovieclipsLength() > 0:
                    self._parse_compressed_movie_clips(cmc)
            except Exception:
                pass  # Neither variant has data
        pos += 4 + mc_size

        # MovieClipModifiers
        mod_size = struct.unpack("<I", inner[pos : pos + 4])[0]
        if mod_size > 0:
            mods = MovieClipModifiers.GetRootAs(
                bytes(inner[pos + 4 : pos + 4 + mod_size]), 0
            )
            for i in range(mods.ModifiersLength()):
                m = mods.Modifiers(i)
                self.modifiers[m.Id()] = m.Type()
        pos += 4 + mod_size

        # Textures
        tex_size = struct.unpack("<I", inner[pos : pos + 4])[0]
        tex = Textures.GetRootAs(
            bytes(inner[pos + 4 : pos + 4 + tex_size]), 0
        )
        for i in range(tex.TexturesLength()):
            tset = tex.Textures(i)
            hr = tset.Highres()
            if hr:
                ext = (
                    hr.ExternalTexture().decode()
                    if hr.ExternalTexture()
                    else None
                )
                self.textures.append(
                    {
                        "width": hr.Width(),
                        "height": hr.Height(),
                        "pixel_type": hr.PixelType(),
                        "external": ext,
                    }
                )

    # ------------------------------------------------------------------
    def _parse_movie_clip(self, clip: Any) -> None:
        """Parse one MovieClip or CompressedMovieClip into internal structures."""
        mc_id = clip.Id()
        children_ids = [clip.ChildrenIds(j) for j in range(clip.ChildrenIdsLength())]
        children = [{"id": cid} for cid in children_ids]
        child_names: list[str] = []
        for j in range(clip.ChildrenNameRefIdsLength()):
            ref = clip.ChildrenNameRefIds(j)
            if 0 < ref <= len(self.strings):
                child_names.append(self.strings[ref - 1])
            else:
                child_names.append("")
        frame_counts: list[int] = []
        frame_labels: list[str] = []
        if clip.FramesLength() > 0:
            for j in range(clip.FramesLength()):
                frame = clip.Frames(j)
                frame_counts.append(frame.UsedTransform())
                lid = frame.LabelRefId()
                label = self.strings[lid - 1] if 0 < lid <= len(self.strings) else ""
                frame_labels.append(label)
        elif hasattr(clip, 'ShortFramesLength') and clip.ShortFramesLength() > 0:
            # Compact variant: uint16 element count, no frame labels
            for j in range(clip.ShortFramesLength()):
                sf = clip.ShortFrames(j)
                frame_counts.append(sf.UsedTransform())
                frame_labels.append("")

        children_blending = [clip.ChildrenBlending(j) for j in range(clip.ChildrenBlendingLength())]
        fps = clip.Framerate() or 24

        # Scaling grid: index into DataStorage.rectangles
        sg: ScalingGrid | None = None
        sgi = clip.ScalingGridIndex()
        if sgi is not None and 0 <= sgi < len(self._scaling_rects):
            sg = self._scaling_rects[sgi]

        fe_offset = clip.FrameElementsOffset()
        if fe_offset is None:
            fe_offset = 0xFFFFFFFF

        self.movie_clip_data[mc_id] = MovieClipData(
            id=mc_id,
            children_ids=children_ids,
            children_names=child_names,
            children_blending=children_blending,
            frame_elements_offset=fe_offset,
            matrix_bank_index=clip.MatrixBankIndex(),
            frame_element_counts=frame_counts,
            frame_labels=frame_labels,
            framerate=fps,
            scaling_grid=sg,
        )

    def _parse_movie_clips(self, mc: Any) -> None:
        """Parse regular MovieClips table."""
        for i in range(mc.MovieclipsLength()):
            self._parse_movie_clip(mc.Movieclips(i))

    def _parse_compressed_movie_clips(self, cmc: Any) -> None:
        """Parse CompressedMovieClips variant.

        CompressedMovieClips share most fields with regular MovieClips
        (children, blending, frames, etc.).  The key difference is that
        frame element data may be stored in a compressed binary buffer
        referenced by ``compressed_data_offset`` rather than in the
        global frame_elements array.  For now we parse the shared
        fields; compressed frame data support can be added later.
        """
        for i in range(cmc.MovieclipsLength()):
            self._parse_movie_clip(cmc.Movieclips(i))

    # ------------------------------------------------------------------
    def get_frame_elements(self, mc_id: int, frame_idx: int = 0) -> list[FrameElement]:
        """Return frame elements for frame *frame_idx* of movie clip *mc_id*."""
        mcd = self.movie_clip_data.get(mc_id)
        if mcd is None or not mcd.frame_element_counts:
            return []
        if mcd.frame_elements_offset == 0xFFFFFFFF:
            return []
        if frame_idx < 0 or frame_idx >= len(mcd.frame_element_counts):
            return []
        fe = self._frame_elements
        # Advance past preceding frames' elements
        off = mcd.frame_elements_offset
        for fi in range(frame_idx):
            off += mcd.frame_element_counts[fi] * 3
        count = mcd.frame_element_counts[frame_idx]
        result: list[FrameElement] = []
        for k in range(count):
            base = off + k * 3
            if base + 2 >= len(fe):
                break
            result.append(FrameElement(
                child_index=int(fe[base]),
                matrix_index=int(fe[base + 1]),
                color_index=int(fe[base + 2]),
            ))
        return result

    def find_frame_by_label(self, mc_id: int, label: str) -> int:
        """Find frame index by label name, returning 0 if not found."""
        mcd = self.movie_clip_data.get(mc_id)
        if mcd is None:
            return 0
        for i, fl in enumerate(mcd.frame_labels):
            if fl == label:
                return i
        return 0

    def get_matrix(self, mc_id: int, matrix_index: int) -> Matrix2x3:
        """Look up a matrix from the appropriate bank for *mc_id*."""
        if matrix_index == _NO_TRANSFORM:
            return Matrix2x3.IDENTITY
        mcd = self.movie_clip_data.get(mc_id)
        bank_idx = mcd.matrix_bank_index if mcd else 0
        if bank_idx < len(self._matrix_banks):
            bank = self._matrix_banks[bank_idx]
            if matrix_index < len(bank):
                return bank[matrix_index]
        return Matrix2x3.IDENTITY

    def get_color(self, mc_id: int, color_index: int) -> ColorTransform:
        """Look up a color transform from the appropriate bank for *mc_id*."""
        if color_index == _NO_TRANSFORM:
            return ColorTransform.IDENTITY
        mcd = self.movie_clip_data.get(mc_id)
        bank_idx = mcd.matrix_bank_index if mcd else 0
        if bank_idx < len(self._color_banks):
            bank = self._color_banks[bank_idx]
            if color_index < len(bank):
                return bank[color_index]
        return ColorTransform.IDENTITY

    # ------------------------------------------------------------------
    def find_shapes_for_export(self, export_name: str) -> list[int]:
        """Return shape-list indices for every shape reachable from *export_name*."""
        mc_id = self.exports.get(export_name)
        if mc_id is None:
            return []
        return self._collect_shapes(mc_id, set())

    def _collect_shapes(self, obj_id: int, visited: set[int]) -> list[int]:
        if obj_id in visited:
            return []
        visited.add(obj_id)
        indices = list(self._shape_id_to_idx.get(obj_id, []))
        mcd = self.movie_clip_data.get(obj_id)
        if mcd:
            for child_id in mcd.children_ids:
                indices.extend(self._collect_shapes(child_id, visited))
        return indices

    # ------------------------------------------------------------------
    def _render_shape(
        self,
        shape_idx: int,
        texture_images: list[Image.Image | None],
        transform: tuple[float, float, float, float, float, float] | None = None,
        _tex_arr_cache: dict[int, np.ndarray] | None = None,
    ) -> tuple[Image.Image | None, float, float]:
        """Render a single shape (all commands), return (image, x_off, y_off).

        When *transform* is given, vertex XY are transformed before
        rasterisation so the output is in the transformed coordinate
        space at full resolution.
        """
        shape = self.shapes[shape_idx]
        parts: list[tuple[Image.Image, float, float, int]] = []
        for cmd in shape["commands"]:
            tex_idx = cmd["texture_index"]
            if tex_idx >= len(texture_images):
                continue
            tex_img = texture_images[tex_idx]
            if tex_img is None:
                continue
            # Reuse cached numpy array for the same texture
            if _tex_arr_cache is not None:
                if tex_idx not in _tex_arr_cache:
                    _tex_arr_cache[tex_idx] = np.array(tex_img)
                cached_arr = _tex_arr_cache[tex_idx]
            else:
                cached_arr = None
            tex = self.textures[tex_idx]
            img, x_off, y_off = render_command(
                cmd["vertices"], tex_img, tex["width"], tex["height"],
                transform=transform, tex_arr=cached_arr,
            )
            if img is None or img.size[0] == 0 or img.size[1] == 0:
                continue
            if img.getchannel("A").getextrema()[1] == 0:
                continue
            parts.append((img, x_off, y_off, 0))
        if not parts:
            return None, 0, 0
        if len(parts) == 1:
            img, x, y, _ = parts[0]
            return img, x, y
        return composite_parts(parts)

    def render_object(
        self,
        obj_id: int,
        texture_images: list[Image.Image | None],
        parent_matrix: Matrix2x3,
        visited: set[int],
        color: ColorTransform | None = None,
        frame_label: str | None = None,
        depth: int = 0,
        blend_mode: int = 0,
        child_labels: dict[int, str] | None = None,
        frame_index: int | None = None,
        ctx: RenderContext | None = None,
        _tex_arr_cache: dict[int, np.ndarray] | None = None,
    ) -> list[tuple[Image.Image, float, float, int]]:
        """Recursively render an object (shape or movie clip) with transforms.

        *frame_label*: if set, child MCs that have a frame with this label
        will render that frame instead of frame 0.
        *blend_mode*: 0=normal, 8=add (additive blending).
        *child_labels*: if set, maps child_index → frame_label for direct
        children.  Only children listed are rendered; others are hidden.
        This is consumed at the first MC level (depth 0) and not propagated
        further - each child uses its assigned label recursively.
        *ctx*: optional render context for custom frame selection, child
        filtering, and mask content injection.
        *frame_index*: if set, use this frame index directly (0-based) for
        the top-level MC only.  Not propagated to children (they use
        *frame_label* or their own default).

        Returns list of (image, global_x, global_y, blend_mode) tuples ready for final compositing.
        """
        if depth > 50:
            return []

        if _tex_arr_cache is None:
            _tex_arr_cache = {}

        rendered: list[tuple[Image.Image, float, float, int]] = []

        # If it's a shape, render it with the parent matrix baked into
        # vertex coordinates so the texture is sampled at full target
        # resolution (no lossy post-rasterisation upscale).
        # Pure translations skip the transform (just offset the result).
        shape_indices = self._shape_id_to_idx.get(obj_id, [])
        _pure_xlate = (
            parent_matrix.a == 1 and parent_matrix.b == 0
            and parent_matrix.c == 0 and parent_matrix.d == 1
        )
        mat_tuple: tuple[float, float, float, float, float, float] | None = None
        if shape_indices and not _pure_xlate and parent_matrix != Matrix2x3.IDENTITY:
            mat_tuple = (
                parent_matrix.a, parent_matrix.b,
                parent_matrix.c, parent_matrix.d,
                parent_matrix.tx, parent_matrix.ty,
            )
        for si in shape_indices:
            img, x_off, y_off = self._render_shape(
                si, texture_images, transform=mat_tuple,
                _tex_arr_cache=_tex_arr_cache,
            )
            if img is None:
                continue
            if color is not None:
                img = color.apply(img)
            if _pure_xlate:
                x_off += parent_matrix.tx
                y_off += parent_matrix.ty
            rendered.append((img, x_off, y_off, blend_mode))

        # If it's a movie clip, process frame elements
        mcd = self.movie_clip_data.get(obj_id)
        if mcd and obj_id not in visited:
            # Branch-scoped visited: siblings may reference the same MC
            # (e.g. three MC 1005 diamond children with different labels)
            # so we must NOT mutate the caller's set.
            child_visited = visited | {obj_id}

            # Pick frame: frame_index > ctx.frame_finder > label match > 0
            frame_idx = 0
            if frame_index is not None:
                frame_idx = frame_index
            elif ctx is not None and ctx.frame_finder is not None and frame_label:
                frame_idx = ctx.frame_finder(obj_id, frame_label)
            elif frame_label:
                frame_idx = self.find_frame_by_label(obj_id, frame_label)

            elements = self.get_frame_elements(obj_id, frame_idx)

            if elements:
                # Optionally limit which children are rendered
                allowed_children = (
                    ctx.render_children.get(obj_id)
                    if ctx is not None and ctx.render_children is not None
                    else None
                )

                # Mask state machine for MovieClipModifiers
                mask_img: Image.Image | None = None
                mask_x: float = 0
                mask_y: float = 0
                capture_mask = False  # next rendered child becomes the mask
                apply_mask = False    # clip children to mask

                for elem in elements:
                    if elem.child_index >= len(mcd.children_ids):
                        continue
                    if allowed_children is not None and elem.child_index not in allowed_children:
                        continue
                    child_id = mcd.children_ids[elem.child_index]

                    # Check if child is a modifier
                    mod_type = self.modifiers.get(child_id)
                    if mod_type == _MOD_MASK:
                        capture_mask = True
                        continue
                    elif mod_type == _MOD_MASKED:
                        apply_mask = True
                        # Inject registered content at the start of the
                        # masked region so that subsequent masked children
                        # (frame overlays) render on top.
                        if (
                            mask_img is not None
                            and ctx is not None
                            and ctx.inject_in_mask is not None
                            and obj_id in ctx.inject_in_mask
                        ):
                            for inj in ctx.inject_in_mask[obj_id]:
                                inj_img, inj_x, inj_y, inj_blend = inj
                                clipped = clip_to_mask(
                                    inj_img, inj_x, inj_y,
                                    mask_img, mask_x, mask_y,
                                )
                                if clipped is not None:
                                    c_img, c_x, c_y = clipped
                                    rendered.append(
                                        (c_img, c_x, c_y, inj_blend)
                                    )
                        continue
                    elif mod_type == _MOD_UNMASKED:
                        apply_mask = False
                        mask_img = None
                        continue

                    # Per-child label override: only render listed
                    # children, each with its own label.
                    if child_labels is not None:
                        if elem.child_index not in child_labels:
                            continue
                        effective_label = child_labels[elem.child_index]
                    else:
                        effective_label = frame_label

                    child_mat = self.get_matrix(obj_id, elem.matrix_index)
                    child_color = self.get_color(obj_id, elem.color_index)
                    combined = parent_matrix @ child_mat
                    child_blend = (
                        mcd.children_blending[elem.child_index]
                        if elem.child_index < len(mcd.children_blending)
                        else 0
                    )

                    # Render child with normal compositing internally
                    child_parts = self.render_object(
                        child_id, texture_images, combined, child_visited,
                        child_color if child_color is not ColorTransform.IDENTITY else color,
                        effective_label, depth + 1,
                        blend_mode=0,
                        ctx=ctx,
                        _tex_arr_cache=_tex_arr_cache,
                    )

                    # If child has non-zero blend, composite fragments into one image
                    # first, then apply the blend to the single result
                    effective_blend = child_blend if child_blend != 0 else blend_mode
                    if effective_blend != 0 and child_parts and len(child_parts) > 1:
                        comp = composite_parts(
                            [(im, x, y, 0) for im, x, y, _ in child_parts]
                        )
                        if comp:
                            c_img, c_x, c_y = comp
                            child_parts = [(c_img, c_x, c_y, effective_blend)]
                        else:
                            child_parts = []
                    elif effective_blend != 0 and child_parts:
                        child_parts = [(im, x, y, effective_blend)
                                       for im, x, y, _ in child_parts]

                    if capture_mask:
                        # Composite this child into a single mask image
                        capture_mask = False
                        if child_parts:
                            mask_result = composite_parts(
                                [(im, x, y, 0) for im, x, y, _ in child_parts]
                            )
                            if mask_result:
                                mask_img, mask_x, mask_y = mask_result
                        continue  # mask shape itself is not drawn

                    if apply_mask and mask_img is not None and child_parts:
                        # Clip each fragment to the mask alpha
                        for cp_img, cp_x, cp_y, cp_blend in child_parts:
                            clipped = clip_to_mask(
                                cp_img, cp_x, cp_y, mask_img, mask_x, mask_y
                            )
                            if clipped is not None:
                                c_img, c_x, c_y = clipped
                                rendered.append((c_img, c_x, c_y, cp_blend))
                    else:
                        rendered.extend(child_parts)

                    # Inject overlay parts after this child (depth 0 only).
                    # Overlays are not subject to the mask state machine
                    # because they represent independent content (e.g. a
                    # frame border) composited at this z-position.
                    if (depth == 0
                            and ctx is not None
                            and ctx.overlay_after_child is not None
                            and elem.child_index in ctx.overlay_after_child):
                        rendered.extend(
                            ctx.overlay_after_child[elem.child_index]
                        )
            elif mcd.frame_elements_offset == 0xFFFFFFFF:
                # No frame element data at all - render children with identity
                for idx, child_id in enumerate(mcd.children_ids):
                    child_blend = (
                        mcd.children_blending[idx]
                        if idx < len(mcd.children_blending)
                        else 0
                    )
                    effective_blend = child_blend if child_blend != 0 else blend_mode
                    child_parts = self.render_object(
                        child_id, texture_images, parent_matrix, child_visited,
                        color, frame_label, depth + 1,
                        blend_mode=0,
                        ctx=ctx,
                        _tex_arr_cache=_tex_arr_cache,
                    )
                    if effective_blend != 0 and child_parts and len(child_parts) > 1:
                        comp = composite_parts(
                            [(im, x, y, 0) for im, x, y, _ in child_parts]
                        )
                        if comp:
                            c_img, c_x, c_y = comp
                            child_parts = [(c_img, c_x, c_y, effective_blend)]
                        else:
                            child_parts = []
                    elif effective_blend != 0 and child_parts:
                        child_parts = [(im, x, y, effective_blend)
                                       for im, x, y, _ in child_parts]
                    rendered.extend(child_parts)
            # else: selected frame explicitly has 0 elements - nothing visible

        return rendered

    # ------------------------------------------------------------------
    def extract_sprite(
        self,
        export_name: str,
        texture_images: list[Image.Image | None],
        output_path: str | Path | None = None,
        frame_label: str | None = None,
        child_labels: dict[int, str] | None = None,
        frame_index: int | None = None,
        ctx: RenderContext | None = None,
    ) -> Image.Image | None:
        """Extract a named sprite, compositing all shapes with correct transforms.

        *frame_label*: if set, child MCs select the frame matching this label
        (e.g. "evo_unlocked") instead of frame 0.
        *child_labels*: if set, maps child_index → frame_label for direct
        children of the export's root MC.  Only children listed in the dict
        are rendered (others are hidden).  Overrides *frame_label* for those
        children; *frame_label* is ignored when *child_labels* is provided.
        *frame_index*: if set, use this frame index (0-based) for the root MC.
        """
        obj_id = self.exports.get(export_name)
        if obj_id is None:
            return None

        rendered = self.render_object(
            obj_id, texture_images, Matrix2x3.IDENTITY, set(),
            frame_label=frame_label,
            child_labels=child_labels,
            frame_index=frame_index,
            ctx=ctx,
        )

        if not rendered:
            return None

        result = composite_parts(rendered)
        if result is None:
            return None

        img, _, _ = result
        if output_path:
            img.save(str(output_path))
        return img

    def extract_sprite_with_offset(
        self,
        export_name: str,
        texture_images: list[Image.Image | None],
        frame_label: str | None = None,
        child_labels: dict[int, str] | None = None,
        frame_index: int | None = None,
        ctx: RenderContext | None = None,
    ) -> tuple[Image.Image, float, float] | None:
        """Like extract_sprite but returns (image, x_offset, y_offset).

        The offsets are in game coordinates (relative to the object origin).
        Multiple exports placed at their offsets will naturally overlay.
        """
        obj_id = self.exports.get(export_name)
        if obj_id is None:
            return None

        rendered = self.render_object(
            obj_id, texture_images, Matrix2x3.IDENTITY, set(),
            frame_label=frame_label,
            child_labels=child_labels,
            frame_index=frame_index,
            ctx=ctx,
        )

        if not rendered:
            return None

        return composite_parts(rendered)

    # ------------------------------------------------------------------
    def get_export_frame_info(self, export_name: str) -> dict[str, Any] | None:
        """Return frame count and labels for an export's root MC."""
        obj_id = self.exports.get(export_name)
        if obj_id is None:
            return None
        mcd = self.movie_clip_data.get(obj_id)
        if mcd is None:
            return None
        return {
            "frame_count": len(mcd.frame_element_counts),
            "frame_labels": mcd.frame_labels,
        }

    # ------------------------------------------------------------------
    def get_shape_bounds(self, shape_idx: int) -> dict[str, Any] | None:
        """Return the UV bounding box for a shape on its texture."""
        shape = self.shapes[shape_idx]
        all_us: list[float] = []
        all_vs: list[float] = []
        tex_idx = None
        for cmd in shape["commands"]:
            tex_idx = cmd["texture_index"]
            if tex_idx >= len(self.textures):
                continue
            tex = self.textures[tex_idx]
            w, h = tex["width"], tex["height"]
            for _x, _y, u, v in cmd["vertices"]:
                all_us.append(u / 65535.0 * w)
                all_vs.append(v / 65535.0 * h)
        if not all_us:
            return None
        return {
            "texture_index": tex_idx,
            "u_min": min(all_us),
            "v_min": min(all_vs),
            "u_max": max(all_us),
            "v_max": max(all_vs),
            "width": max(all_us) - min(all_us),
            "height": max(all_vs) - min(all_vs),
        }


# ======================================================================
# Helpers
# ======================================================================

def clip_to_mask(
    img: Image.Image,
    img_x: float,
    img_y: float,
    mask: Image.Image,
    mask_x: float,
    mask_y: float,
) -> tuple[Image.Image, float, float] | None:
    """Clip *img* to the alpha channel of *mask*, both in global coordinates."""
    # Find overlap region
    ix1, iy1 = int(img_x), int(img_y)
    ix2, iy2 = ix1 + img.width, iy1 + img.height
    mx1, my1 = int(mask_x), int(mask_y)
    mx2, my2 = mx1 + mask.width, my1 + mask.height

    ox1 = max(ix1, mx1)
    oy1 = max(iy1, my1)
    ox2 = min(ix2, mx2)
    oy2 = min(iy2, my2)

    if ox1 >= ox2 or oy1 >= oy2:
        return None

    # Crop both to the overlap region
    img_crop = img.crop((ox1 - ix1, oy1 - iy1, ox2 - ix1, oy2 - iy1))
    mask_crop = mask.crop((ox1 - mx1, oy1 - my1, ox2 - mx1, oy2 - my1))

    # Multiply img alpha by mask alpha
    img_arr = np.array(img_crop).copy()
    mask_alpha = np.array(mask_crop)[:, :, 3].astype(np.uint16)
    img_arr[:, :, 3] = (img_arr[:, :, 3].astype(np.uint16) * mask_alpha // 255).astype(np.uint8)

    if img_arr[:, :, 3].max() == 0:
        return None

    return Image.fromarray(img_arr), float(ox1), float(oy1)


def composite_parts(
    parts: list[tuple[Image.Image, float, float, int]],
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[Image.Image, float, float] | None:
    """Composite multiple (image, x, y, blend_mode) fragments into one image.

    *bounds* – optional ``(xmin, ymin, xmax, ymax)`` canvas extents.
    When provided the canvas is sized to these bounds instead of being
    derived from the parts.  Bounds are snapped to integer pixels
    (``floor`` for min, ``ceil`` for max) so that part positions round
    consistently regardless of sub-pixel shifts in animated content.

    Returns ``(image, xmin, ymin)`` where *xmin*/*ymin* are the
    integer-snapped canvas origin, or *None* if the parts list is empty.

    Supports all 14 Flash blend modes (0=Normal through 14=HardLight).
    """
    if not parts:
        return None

    if bounds is not None:
        all_x_min = math.floor(bounds[0])
        all_y_min = math.floor(bounds[1])
        all_x_max = math.ceil(bounds[2])
        all_y_max = math.ceil(bounds[3])
    else:
        all_x_min = math.floor(min(xo for _, xo, _, _ in parts))
        all_y_min = math.floor(min(yo for _, _, yo, _ in parts))
        all_x_max = math.ceil(max(xo + im.width for im, xo, _, _ in parts))
        all_y_max = math.ceil(max(yo + im.height for im, _, yo, _ in parts))

    out_w = all_x_max - all_x_min
    out_h = all_y_max - all_y_min
    if out_w <= 0 or out_h <= 0 or out_w > 8192 or out_h > 8192:
        return None

    result = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
    for img, xo, yo, blend in parts:
        px = round(xo) - all_x_min
        py = round(yo) - all_y_min
        if blend == 0:
            result.alpha_composite(img, (px, py))
        else:
            blend_layer(result, img, px, py, blend)

    return result, float(all_x_min), float(all_y_min)


def additive_blend(
    base: Image.Image,
    overlay: Image.Image,
    px: int,
    py: int,
) -> None:
    """In-place additive blend of overlay onto base at (px, py).

    Kept for backward compatibility.  Delegates to :func:`blend_layer`.
    """
    blend_layer(base, overlay, px, py, 8)


def _clip_regions(
    base: Image.Image, overlay: Image.Image, px: int, py: int,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int] | None:
    """Compute clipped base/overlay array slices. Returns None if no overlap."""
    ow, oh = overlay.size
    bw, bh = base.size
    x1, y1 = max(px, 0), max(py, 0)
    x2, y2 = min(px + ow, bw), min(py + oh, bh)
    if x1 >= x2 or y1 >= y2:
        return None
    return (
        np.array(base)[y1:y2, x1:x2],
        np.array(overlay)[y1 - py : y2 - py, x1 - px : x2 - px],
        x1, y1, x2, y2,
    )


def blend_layer(
    base: Image.Image,
    overlay: Image.Image,
    px: int,
    py: int,
    blend_mode: int,
) -> None:
    """In-place blend *overlay* onto *base* at (px, py) using *blend_mode*.

    Implements all 14 Adobe Flash blend modes used by Supercell:

    ====  ===========  ==========================================
    Code  Name         Formula (Cb=base, Cs=source, per channel)
    ====  ===========  ==========================================
    0     Normal       Standard alpha composite (use alpha_composite instead)
    2     Layer        Same as Normal (group isolation hint only)
    3     Multiply     Cb × Cs
    4     Screen       1 − (1−Cb)(1−Cs)
    5     Lighten      max(Cb, Cs)
    6     Darken       min(Cb, Cs)
    7     Difference   |Cb − Cs|
    8     Add          Cb + Cs (clamped)
    9     Subtract     Cb − Cs (clamped ≥ 0)
    10    Invert       1 − Cb (ignores source RGB)
    11    Alpha        Uses source alpha only (no RGB change)
    12    Erase        Removes base alpha where source has alpha
    13    Overlay      if Cb<½: 2·Cb·Cs  else: 1−2·(1−Cb)·(1−Cs)
    14    HardLight    if Cs<½: 2·Cb·Cs  else: 1−2·(1−Cb)·(1−Cs)
    ====  ===========  ==========================================

    All formulas operate on pre-alpha-mixed values in [0, 255].
    Source alpha controls how much the blend result replaces the base:
    ``result = base + (blended − base) × src_alpha / 255``
    """
    regions = _clip_regions(base, overlay, px, py)
    if regions is None:
        return
    b_slice, o_slice, x1, y1, x2, y2 = regions

    b = b_slice.astype(np.int32)
    o = o_slice.astype(np.int32)
    br, bg, bb, ba = b[:,:,0], b[:,:,1], b[:,:,2], b[:,:,3]
    sr, sg, sb, sa = o[:,:,0], o[:,:,1], o[:,:,2], o[:,:,3]

    if blend_mode == 8:  # Add
        # For additive, alpha = max so glow is visible in extraction
        out_a = np.maximum(ba, sa)
        # Apply source alpha weighting to RGB
        out_r = np.minimum(br + sr * sa // 255, 255)
        out_g = np.minimum(bg + sg * sa // 255, 255)
        out_b = np.minimum(bb + sb * sa // 255, 255)
        result = np.stack([out_r, out_g, out_b, out_a], axis=2)
        full = np.array(base)
        full[y1:y2, x1:x2] = np.clip(result, 0, 255).astype(np.uint8)
        base.paste(Image.fromarray(full))
        return

    if blend_mode in (0, 2):  # Normal / Layer
        # Layer (2) is a group isolation hint; compositing is identical.
        base.alpha_composite(overlay, (px, py))
        return

    if blend_mode == 11:  # Alpha — only transfers alpha, no RGB
        out_a = np.minimum(ba + sa * (255 - ba) // 255, 255)
        full = np.array(base)
        full[y1:y2, x1:x2, 3] = np.clip(out_a, 0, 255).astype(np.uint8)
        base.paste(Image.fromarray(full))
        return

    if blend_mode == 12:  # Erase — removes alpha where source has alpha
        out_a = ba * (255 - sa) // 255
        full = np.array(base)
        full[y1:y2, x1:x2, 3] = np.clip(out_a, 0, 255).astype(np.uint8)
        base.paste(Image.fromarray(full))
        return

    # Standard RGB blend modes: compute blended RGB, then mix by source alpha
    if blend_mode == 3:  # Multiply
        cr = br * sr // 255
        cg = bg * sg // 255
        cb = bb * sb // 255
    elif blend_mode == 4:  # Screen
        cr = br + sr - br * sr // 255
        cg = bg + sg - bg * sg // 255
        cb = bb + sb - bb * sb // 255
    elif blend_mode == 5:  # Lighten
        cr = np.maximum(br, sr)
        cg = np.maximum(bg, sg)
        cb = np.maximum(bb, sb)
    elif blend_mode == 6:  # Darken
        cr = np.minimum(br, sr)
        cg = np.minimum(bg, sg)
        cb = np.minimum(bb, sb)
    elif blend_mode == 7:  # Difference
        cr = np.abs(br - sr)
        cg = np.abs(bg - sg)
        cb = np.abs(bb - sb)
    elif blend_mode == 9:  # Subtract
        cr = np.maximum(br - sr, 0)
        cg = np.maximum(bg - sg, 0)
        cb = np.maximum(bb - sb, 0)
    elif blend_mode == 10:  # Invert (ignores source RGB, flips base)
        cr = 255 - br
        cg = 255 - bg
        cb = 255 - bb
    elif blend_mode == 13:  # Overlay
        cr = np.where(br < 128, 2 * br * sr // 255, 255 - 2 * (255 - br) * (255 - sr) // 255)
        cg = np.where(bg < 128, 2 * bg * sg // 255, 255 - 2 * (255 - bg) * (255 - sg) // 255)
        cb = np.where(bb < 128, 2 * bb * sb // 255, 255 - 2 * (255 - bb) * (255 - sb) // 255)
    elif blend_mode == 14:  # HardLight (Overlay with source/base swapped)
        cr = np.where(sr < 128, 2 * br * sr // 255, 255 - 2 * (255 - br) * (255 - sr) // 255)
        cg = np.where(sg < 128, 2 * bg * sg // 255, 255 - 2 * (255 - bg) * (255 - sg) // 255)
        cb = np.where(sb < 128, 2 * bb * sb // 255, 255 - 2 * (255 - bb) * (255 - sb) // 255)
    else:
        # Unknown blend mode — fall back to normal alpha composite
        base.alpha_composite(overlay, (px, py))
        return

    # Mix blended result with base by source alpha:
    # out = base + (blended - base) * src_alpha / 255
    out_r = br + (cr - br) * sa // 255
    out_g = bg + (cg - bg) * sa // 255
    out_b = bb + (cb - bb) * sa // 255
    # Alpha: standard "over" operator
    out_a = ba + sa * (255 - ba) // 255

    result = np.stack([out_r, out_g, out_b, out_a], axis=2)
    full = np.array(base)
    full[y1:y2, x1:x2] = np.clip(result, 0, 255).astype(np.uint8)
    base.paste(Image.fromarray(full))

