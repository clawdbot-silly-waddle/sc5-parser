"""SC v5 file parser.

Decodes the binary SC v5 container (header → FileDescriptor FlatBuffer →
ZSTD-compressed inner stream) and exposes shapes, movie-clips, textures
and named exports for downstream extraction.
"""

from __future__ import annotations

import struct
import warnings
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
from sc5_parser.models import (
    ColorTransform,
    FrameElement,
    Matrix2x3,
    MovieClipData,
    RenderContext,
    ScalingGrid,
    ShapeDict,
    TextFieldData,
)

# Sentinel: "no matrix" / "no color transform" in frame elements
_NO_TRANSFORM = 0xFFFF

# Sentinel: movie clip has no frame element data at all
_NO_FRAME_ELEMENTS = 0xFFFFFFFF

# Maximum decompressed size for ZSTD streams (100 MiB)
_MAX_DECOMPRESS_SIZE = 100 * 1024 * 1024

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
        self.shape_id_to_idx: dict[int, list[int]] = {}
        self._frame_elements: np.ndarray | None = None
        self.matrix_banks: list[list[Matrix2x3]] = []
        self.color_banks: list[list[ColorTransform]] = []
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
            comp_data, max_output_size=_MAX_DECOMPRESS_SIZE
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
            self.matrix_banks.append(matrices)
            colors: list[ColorTransform] = []
            for ci in range(bank_fb.ColorsLength()):
                c = bank_fb.Colors(ci)
                colors.append(ColorTransform(
                    r_mul=c.RMul(), g_mul=c.GMul(), b_mul=c.BMul(),
                    alpha=c.Alpha(),
                    r_add=c.RAdd(), g_add=c.GAdd(), b_add=c.BAdd(),
                ))
            self.color_banks.append(colors)

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
                    self.matrix_banks.append(ext_matrices)
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
                    self.color_banks.append(ext_colors)
            except Exception:
                warnings.warn(
                    f"Failed to parse external matrix/color banks in {self.sc_path}",
                    stacklevel=2,
                )

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
            self.shape_id_to_idx.setdefault(sid, []).append(
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
                warnings.warn(
                    f"Failed to parse compressed movie clips in {self.sc_path}",
                    stacklevel=2,
                )
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
            fe_offset = _NO_FRAME_ELEMENTS

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
        if mcd.frame_elements_offset == _NO_FRAME_ELEMENTS:
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
        if bank_idx < len(self.matrix_banks):
            bank = self.matrix_banks[bank_idx]
            if matrix_index < len(bank):
                return bank[matrix_index]
        return Matrix2x3.IDENTITY

    def get_color(self, mc_id: int, color_index: int) -> ColorTransform:
        """Look up a color transform from the appropriate bank for *mc_id*."""
        if color_index == _NO_TRANSFORM:
            return ColorTransform.IDENTITY
        mcd = self.movie_clip_data.get(mc_id)
        bank_idx = mcd.matrix_bank_index if mcd else 0
        if bank_idx < len(self.color_banks):
            bank = self.color_banks[bank_idx]
            if color_index < len(bank):
                return bank[color_index]
        return ColorTransform.IDENTITY


    # ------------------------------------------------------------------
    # Rendering — delegated to sc5_parser.render module
    # ------------------------------------------------------------------

    def find_shapes_for_export(self, export_name: str) -> list[int]:
        """Return shape-list indices for every shape reachable from *export_name*."""
        from sc5_parser.render import find_shapes_for_export
        return find_shapes_for_export(self, export_name)

    def _collect_shapes(self, obj_id: int, visited: set[int]) -> list[int]:
        from sc5_parser.render import collect_shapes
        return collect_shapes(self, obj_id, visited)

    def _render_shape(
        self,
        shape_idx: int,
        texture_images: list[Image.Image | None],
        transform: tuple[float, float, float, float, float, float] | None = None,
        _tex_arr_cache: dict[int, np.ndarray] | None = None,
    ) -> tuple[Image.Image | None, float, float]:
        """Render a single shape (all commands), return (image, x_off, y_off)."""
        from sc5_parser.render import render_shape
        return render_shape(self, shape_idx, texture_images, transform, _tex_arr_cache)

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
        """Recursively render an object (shape or movie clip) with transforms."""
        from sc5_parser.render import render_object
        return render_object(
            self, obj_id, texture_images, parent_matrix, visited,
            color, frame_label, depth, blend_mode, child_labels,
            frame_index, ctx, _tex_arr_cache,
        )

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
        """Extract a named sprite, compositing all shapes with correct transforms."""
        from sc5_parser.render import extract_sprite
        return extract_sprite(
            self, export_name, texture_images, output_path,
            frame_label, child_labels, frame_index, ctx,
        )

    def extract_sprite_with_offset(
        self,
        export_name: str,
        texture_images: list[Image.Image | None],
        frame_label: str | None = None,
        child_labels: dict[int, str] | None = None,
        frame_index: int | None = None,
        ctx: RenderContext | None = None,
    ) -> tuple[Image.Image, float, float] | None:
        """Like extract_sprite but returns (image, x_offset, y_offset)."""
        from sc5_parser.render import extract_sprite_with_offset
        return extract_sprite_with_offset(
            self, export_name, texture_images, frame_label,
            child_labels, frame_index, ctx,
        )

    def get_export_frame_info(self, export_name: str) -> dict[str, Any] | None:
        """Return frame count and labels for an export's root MC."""
        from sc5_parser.render import get_export_frame_info
        return get_export_frame_info(self, export_name)

    def get_shape_bounds(self, shape_idx: int) -> dict[str, Any] | None:
        """Return the UV bounding box for a shape on its texture."""
        from sc5_parser.render import get_shape_bounds
        return get_shape_bounds(self, shape_idx)
