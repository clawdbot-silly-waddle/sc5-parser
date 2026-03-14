"""SC v5 file parser.

Decodes the binary SC v5 container (header → FileDescriptor FlatBuffer →
ZSTD-compressed inner stream) and exposes shapes, movie-clips, textures
and named exports for downstream extraction.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import zstandard
from PIL import Image

from sc5_parser._schemas.sc.flash.SC2.DataStorage import DataStorage
from sc5_parser._schemas.sc.flash.SC2.ExportNames import ExportNames
from sc5_parser._schemas.sc.flash.SC2.FileDescriptor import FileDescriptor
from sc5_parser._schemas.sc.flash.SC2.MovieClipModifiers import MovieClipModifiers
from sc5_parser._schemas.sc.flash.SC2.MovieClips import MovieClips
from sc5_parser._schemas.sc.flash.SC2.Shapes import Shapes
from sc5_parser._schemas.sc.flash.SC2.Textures import Textures
from sc5_parser.renderer import render_command

ShapeDict = dict[str, Any]

# Sentinel: "no matrix" / "no color transform" in frame elements
_NO_TRANSFORM = 0xFFFF

# MovieClipModifier types
_MOD_MASK = 38      # Defines the start of a mask group; next child is the mask shape
_MOD_MASKED = 39    # Children after this are clipped by the mask
_MOD_UNMASKED = 40  # End of masked group; children render normally


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


class SC5File:
    """Parsed representation of an SC v5 file."""

    def __init__(self, sc_path: str | Path) -> None:
        self.sc_path = Path(sc_path)
        self.shapes: list[ShapeDict] = []
        self.exports: dict[str, int] = {}  # name → movie-clip / shape id
        self.textures: list[dict[str, Any]] = []
        self.movie_clips: dict[int, dict[str, Any]] = {}
        self.movie_clip_data: dict[int, MovieClipData] = {}
        self.modifiers: dict[int, int] = {}  # id → modifier type (38/39/40)
        self.strings: list[str] = []
        self.vertices: list[tuple[float, float, int, int]] = []
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

        comp_start = 10 + fd_size
        comp_data = (
            raw[comp_start : comp_start + fd.CompressedSize()]
            if fd.CompressedSize()
            else raw[comp_start:]
        )

        inner = zstandard.ZstdDecompressor().decompress(
            comp_data, max_output_size=100 * 1024 * 1024
        )

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

        # --- Matrix banks --------------------------------------------------
        for bi in range(ds.MatrixBanksLength()):
            bank_fb = ds.MatrixBanks(bi)
            matrices: list[Matrix2x3] = []
            for mi in range(bank_fb.MatricesLength()):
                m = bank_fb.Matrices(mi)
                matrices.append(
                    Matrix2x3(a=m.A(), b=m.B(), c=m.C(), d=m.D(), tx=m.Tx(), ty=m.Ty())
                )
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

        # TextFields (skip)
        tf_size = struct.unpack("<I", inner[pos : pos + 4])[0]
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

        # MovieClips
        mc_size = struct.unpack("<I", inner[pos : pos + 4])[0]
        mc = MovieClips.GetRootAs(
            bytes(inner[pos + 4 : pos + 4 + mc_size]), 0
        )
        for i in range(mc.MovieclipsLength()):
            clip = mc.Movieclips(i)
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
            # Frame element counts and labels per frame
            frame_counts: list[int] = []
            frame_labels: list[str] = []
            for j in range(clip.FramesLength()):
                frame = clip.Frames(j)
                frame_counts.append(frame.UsedTransform())
                lid = frame.LabelRefId()
                label = self.strings[lid - 1] if 0 < lid <= len(self.strings) else ""
                frame_labels.append(label)

            self.movie_clips[mc_id] = {
                "id": mc_id,
                "children": children,
                "children_names": child_names,
                "frame_count": clip.FramesLength(),
            }
            children_blending = [clip.ChildrenBlending(j) for j in range(clip.ChildrenBlendingLength())]
            self.movie_clip_data[mc_id] = MovieClipData(
                id=mc_id,
                children_ids=children_ids,
                children_names=child_names,
                children_blending=children_blending,
                frame_elements_offset=clip.FrameElementsOffset(),
                matrix_bank_index=clip.MatrixBankIndex(),
                frame_element_counts=frame_counts,
                frame_labels=frame_labels,
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
    def _get_frame_elements(self, mc_id: int, frame_idx: int = 0) -> list[FrameElement]:
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

    def _find_frame_by_label(self, mc_id: int, label: str) -> int:
        """Find frame index by label name, returning 0 if not found."""
        mcd = self.movie_clip_data.get(mc_id)
        if mcd is None:
            return 0
        for i, fl in enumerate(mcd.frame_labels):
            if fl == label:
                return i
        return 0

    def _get_matrix(self, mc_id: int, matrix_index: int) -> Matrix2x3:
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

    def _get_color(self, mc_id: int, color_index: int) -> ColorTransform:
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
        mc = self.movie_clips.get(obj_id)
        if mc:
            for child in mc["children"]:
                indices.extend(self._collect_shapes(child["id"], visited))
        return indices

    # ------------------------------------------------------------------
    def _render_shape(
        self,
        shape_idx: int,
        texture_images: list[Image.Image | None],
    ) -> tuple[Image.Image | None, float, float]:
        """Render a single shape (all commands), return (image, x_off, y_off)."""
        shape = self.shapes[shape_idx]
        parts: list[tuple[Image.Image, float, float, int]] = []
        for cmd in shape["commands"]:
            tex_idx = cmd["texture_index"]
            if tex_idx >= len(texture_images):
                continue
            tex_img = texture_images[tex_idx]
            if tex_img is None:
                continue
            tex = self.textures[tex_idx]
            img, x_off, y_off = render_command(
                cmd["vertices"], tex_img, tex["width"], tex["height"]
            )
            if img is None or img.size[0] == 0 or img.size[1] == 0:
                continue
            if np.array(img)[:, :, 3].max() == 0:
                continue
            parts.append((img, x_off, y_off, 0))
        if not parts:
            return None, 0, 0
        if len(parts) == 1:
            img, x, y, _ = parts[0]
            return img, x, y
        return _composite_parts(parts)

    def _render_object(
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
    ) -> list[tuple[Image.Image, float, float, int]]:
        """Recursively render an object (shape or movie clip) with transforms.

        *frame_label*: if set, child MCs that have a frame with this label
        will render that frame instead of frame 0.
        *blend_mode*: 0=normal, 8=add (additive blending).
        *child_labels*: if set, maps child_index → frame_label for direct
        children.  Only children listed are rendered; others are hidden.
        This is consumed at the first MC level (depth 0) and not propagated
        further — each child uses its assigned label recursively.
        *frame_index*: if set, use this frame index directly (0-based) for
        the top-level MC only.  Not propagated to children (they use
        *frame_label* or their own default).

        Returns list of (image, global_x, global_y, blend_mode) tuples ready for final compositing.
        """
        if depth > 50:
            return []

        rendered: list[tuple[Image.Image, float, float, int]] = []

        # If it's a shape, render it and apply the parent matrix
        shape_indices = self._shape_id_to_idx.get(obj_id, [])
        for si in shape_indices:
            img, x_off, y_off = self._render_shape(si, texture_images)
            if img is None:
                continue
            if color is not None:
                img = color.apply(img)
            transformed = _apply_matrix(img, x_off, y_off, parent_matrix)
            if transformed is not None:
                t_img, t_x, t_y = transformed
                rendered.append((t_img, t_x, t_y, blend_mode))

        # If it's a movie clip, process frame elements
        mcd = self.movie_clip_data.get(obj_id)
        if mcd and obj_id not in visited:
            visited.add(obj_id)

            # Pick frame: frame_index > label match > 0
            frame_idx = 0
            if frame_index is not None:
                frame_idx = frame_index
            elif frame_label:
                frame_idx = self._find_frame_by_label(obj_id, frame_label)

            elements = self._get_frame_elements(obj_id, frame_idx)

            if elements:
                # Optionally limit which children are rendered (e.g.
                # to suppress sparkle shapes from glow MCs).
                _render_children = getattr(self, '_render_children', None)
                allowed_children = (
                    _render_children.get(obj_id)
                    if _render_children else None
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

                    child_mat = self._get_matrix(obj_id, elem.matrix_index)
                    child_color = self._get_color(obj_id, elem.color_index)
                    combined = parent_matrix @ child_mat
                    child_blend = (
                        mcd.children_blending[elem.child_index]
                        if elem.child_index < len(mcd.children_blending)
                        else 0
                    )

                    # Render child with normal compositing internally
                    child_parts = self._render_object(
                        child_id, texture_images, combined, visited,
                        child_color if child_color is not ColorTransform.IDENTITY else color,
                        effective_label, depth + 1,
                        blend_mode=0,
                    )

                    # If child has non-zero blend, composite fragments into one image
                    # first, then apply the blend to the single result
                    effective_blend = child_blend if child_blend != 0 else blend_mode
                    if effective_blend != 0 and child_parts and len(child_parts) > 1:
                        comp = _composite_parts(
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
                            mask_result = _composite_parts(
                                [(im, x, y, 0) for im, x, y, _ in child_parts]
                            )
                            if mask_result:
                                mask_img, mask_x, mask_y = mask_result
                        continue  # mask shape itself is not drawn

                    if apply_mask and mask_img is not None and child_parts:
                        # Clip each fragment to the mask alpha
                        for cp_img, cp_x, cp_y, cp_blend in child_parts:
                            clipped = _clip_to_mask(
                                cp_img, cp_x, cp_y, mask_img, mask_x, mask_y
                            )
                            if clipped is not None:
                                c_img, c_x, c_y = clipped
                                rendered.append((c_img, c_x, c_y, cp_blend))
                    else:
                        rendered.extend(child_parts)
            elif mcd.frame_elements_offset == 0xFFFFFFFF:
                # No frame element data at all — render children with identity
                for idx, child_id in enumerate(mcd.children_ids):
                    child_blend = (
                        mcd.children_blending[idx]
                        if idx < len(mcd.children_blending)
                        else 0
                    )
                    effective_blend = child_blend if child_blend != 0 else blend_mode
                    child_parts = self._render_object(
                        child_id, texture_images, parent_matrix, visited,
                        color, frame_label, depth + 1,
                        blend_mode=0,
                    )
                    if effective_blend != 0 and child_parts and len(child_parts) > 1:
                        comp = _composite_parts(
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
            # else: selected frame explicitly has 0 elements — nothing visible
            visited.discard(obj_id)

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

        rendered = self._render_object(
            obj_id, texture_images, Matrix2x3.IDENTITY, set(),
            frame_label=frame_label,
            child_labels=child_labels,
            frame_index=frame_index,
        )

        if not rendered:
            return None

        result = _composite_parts(rendered)
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
    ) -> tuple[Image.Image, float, float] | None:
        """Like extract_sprite but returns (image, x_offset, y_offset).

        The offsets are in game coordinates (origin = center of card).
        Multiple exports placed at their offsets will naturally overlay.
        """
        obj_id = self.exports.get(export_name)
        if obj_id is None:
            return None

        rendered = self._render_object(
            obj_id, texture_images, Matrix2x3.IDENTITY, set(),
            frame_label=frame_label,
            child_labels=child_labels,
            frame_index=frame_index,
        )

        if not rendered:
            return None

        return _composite_parts(rendered)

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

def _clip_to_mask(
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


def _apply_matrix(
    img: Image.Image,
    x_off: float,
    y_off: float,
    mat: Matrix2x3,
) -> tuple[Image.Image, float, float] | None:
    """Apply an affine matrix to a rendered sprite fragment.

    Takes the fragment at local position (x_off, y_off) and transforms it.
    Returns (transformed_image, new_x, new_y) in the parent coordinate space.
    """
    if mat.a == 1 and mat.b == 0 and mat.c == 0 and mat.d == 1:
        # Pure translation — skip expensive affine transform
        return img, x_off + mat.tx, y_off + mat.ty

    w, h = img.size

    # Transform the four corners to find the output bounding box
    corners = [
        (x_off, y_off),
        (x_off + w, y_off),
        (x_off, y_off + h),
        (x_off + w, y_off + h),
    ]
    txs = [mat.a * cx + mat.b * cy + mat.tx for cx, cy in corners]
    tys = [mat.c * cx + mat.d * cy + mat.ty for cx, cy in corners]

    out_x_min = min(txs)
    out_y_min = min(tys)
    out_x_max = max(txs)
    out_y_max = max(tys)

    out_w = max(1, int(out_x_max - out_x_min + 1.5))
    out_h = max(1, int(out_y_max - out_y_min + 1.5))

    if out_w > 4096 or out_h > 4096:
        return None

    # PIL affine transform uses the INVERSE matrix:
    # for each output pixel (ox, oy), find input pixel (ix, iy)
    # We need: (ix - x_off, iy - y_off) in the source image
    # where (ix, iy) = inv(mat) @ (ox + out_x_min, oy + out_y_min)
    det = mat.a * mat.d - mat.b * mat.c
    if abs(det) < 1e-10:
        return None
    inv_a = mat.d / det
    inv_b = -mat.b / det
    inv_c = -mat.c / det
    inv_d = mat.a / det
    inv_tx = (mat.b * mat.ty - mat.d * mat.tx) / det
    inv_ty = (mat.c * mat.tx - mat.a * mat.ty) / det

    # PIL transform coefficients map output (ox, oy) → input (ix, iy):
    # ix = a*ox + b*oy + c
    # iy = d*ox + e*oy + f
    # We need to account for the output offset (out_x_min, out_y_min) and
    # the source image offset (x_off, y_off).
    coeffs = (
        inv_a,
        inv_b,
        inv_a * out_x_min + inv_b * out_y_min + inv_tx - x_off,
        inv_c,
        inv_d,
        inv_c * out_x_min + inv_d * out_y_min + inv_ty - y_off,
    )

    result = img.transform(
        (out_w, out_h), Image.AFFINE, coeffs, Image.BILINEAR
    )
    return result, out_x_min, out_y_min


def _composite_parts(
    parts: list[tuple[Image.Image, float, float, int]],
) -> tuple[Image.Image, float, float] | None:
    """Composite multiple (image, x, y, blend_mode) fragments into one image.

    Blend modes: 0=normal (alpha composite), 8=add (additive).
    """
    if not parts:
        return None

    all_x_min = min(xo for _, xo, _, _ in parts)
    all_y_min = min(yo for _, _, yo, _ in parts)
    all_x_max = max(xo + im.width for im, xo, _, _ in parts)
    all_y_max = max(yo + im.height for im, _, yo, _ in parts)

    out_w = int(all_x_max - all_x_min + 0.5)
    out_h = int(all_y_max - all_y_min + 0.5)
    if out_w <= 0 or out_h <= 0 or out_w > 8192 or out_h > 8192:
        return None

    result = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
    for img, xo, yo, blend in parts:
        px = int(xo - all_x_min)
        py = int(yo - all_y_min)
        if blend == 8:
            # Additive blend: add RGB weighted by overlay alpha, keep base alpha
            _additive_blend(result, img, px, py)
        else:
            result.alpha_composite(img, (px, py))

    return result, all_x_min, all_y_min


def _additive_blend(
    base: Image.Image,
    overlay: Image.Image,
    px: int,
    py: int,
) -> None:
    """In-place additive blend of overlay onto base at (px, py).

    Additive blend adds brightness to existing content without introducing new
    opacity.  base_rgb += overlay_rgb * overlay_alpha / 255; alpha stays as-is.
    """
    ow, oh = overlay.size
    bw, bh = base.size
    # Clip to base bounds
    x1, y1 = max(px, 0), max(py, 0)
    x2, y2 = min(px + ow, bw), min(py + oh, bh)
    if x1 >= x2 or y1 >= y2:
        return

    base_arr = np.array(base)
    over_arr = np.array(overlay)

    # Slices in base and overlay coordinate systems
    bslice = base_arr[y1:y2, x1:x2].astype(np.uint16)
    oslice = over_arr[y1 - py : y2 - py, x1 - px : x2 - px].astype(np.uint16)

    alpha = oslice[:, :, 3:4]  # (h, w, 1) broadcast
    # Add RGB weighted by overlay alpha
    bslice[:, :, :3] = np.minimum(bslice[:, :, :3] + oslice[:, :, :3] * alpha // 255, 255)
    # Alpha: take max so additive content is visible in extraction
    bslice[:, :, 3] = np.maximum(bslice[:, :, 3], oslice[:, :, 3])

    base_arr[y1:y2, x1:x2] = bslice.astype(np.uint8)
    base.paste(Image.fromarray(base_arr))


# ======================================================================
# Champion card compositor
# ======================================================================

# MC 1008 child indices for card_frame_champion:
_CARD_CHILD_NOTCH_BASE = 0  # MC 982: full-width notch
_CARD_CHILD_NOTCH_RIGHT = 1  # MC 987: right half overlay
_CARD_CHILD_NOTCH_LEFT = 2  # MC 988: left half overlay
_CARD_CHILD_GLOW = 3  # MC 1001: frame glow border + mask
_CARD_CHILD_DIAMOND_RIGHT = 5  # MC 1005 @ tx=18.5
_CARD_CHILD_DIAMOND_LEFT = 6  # MC 1005 @ tx=-18.6

# Glow sub-MCs (990=evo, 1000=hero) have 34 frames. Frames 0-3 are
# empty, 4-32 show particles/clouds, and frame 33 is a clean border
# with just the outline shape and clipping mask (no particle effects).
_GLOW_CLEAN_FRAME = 33

# Inner glow MCs (849=evo, 973=hero) always include sparkle shapes in
# every frame.  For a static card render we only want child 0 (the frame
# border Shape 397) — sparkle edges create visible artifacts.
_GLOW_BORDER_ONLY: dict[int, frozenset[int]] = {
    849: frozenset({0}),
    973: frozenset({0}),
}

# Shape 392 is the card portrait clipping mask — a solid rounded rectangle
# matching the interior of the champion frame border.
_CARD_PORTRAIT_MASK_SHAPE = 392


def _mask_hierarchy_transform(
    card_sc: SC5File,
    card_mc_id: int,
    frame_finder: Callable[[int, str], int],
) -> Matrix2x3:
    """Compute the accumulated transform from the card MC to Shape 392.

    Traces: card MC → child[_CARD_CHILD_GLOW] (MC 1001) → inner glow MC
    (MC 990/1000) → Shape 392, multiplying matrices along the path.
    """
    result = Matrix2x3.IDENTITY
    card_mcd = card_sc.movie_clip_data.get(card_mc_id)
    if card_mcd is None:
        return result

    # Step 1: card MC → glow child
    card_fe = card_sc._get_frame_elements(card_mc_id, 0)
    glow_mc_id: int | None = None
    for elem in card_fe:
        if elem.child_index == _CARD_CHILD_GLOW:
            result = result @ card_sc._get_matrix(card_mc_id, elem.matrix_index)
            glow_mc_id = card_mcd.children_ids[_CARD_CHILD_GLOW]
            break
    if glow_mc_id is None:
        return result

    # Step 2: glow MC → inner glow MC (990 or 1000)
    glow_mcd = card_sc.movie_clip_data.get(glow_mc_id)
    if glow_mcd is None:
        return result
    glow_frame = frame_finder(glow_mc_id, "")
    glow_fe = card_sc._get_frame_elements(glow_mc_id, glow_frame)
    if not glow_fe:
        return result
    result = result @ card_sc._get_matrix(glow_mc_id, glow_fe[0].matrix_index)
    inner_mc_id = glow_mcd.children_ids[glow_fe[0].child_index]

    # Step 3: inner glow MC → Shape 392
    inner_mcd = card_sc.movie_clip_data.get(inner_mc_id)
    if inner_mcd is None:
        return result
    inner_frame = min(_GLOW_CLEAN_FRAME, len(inner_mcd.frame_element_counts) - 1)
    inner_fe = card_sc._get_frame_elements(inner_mc_id, inner_frame)
    for ie in inner_fe:
        if (ie.child_index < len(inner_mcd.children_ids)
                and inner_mcd.children_ids[ie.child_index]
                == _CARD_PORTRAIT_MASK_SHAPE):
            result = result @ card_sc._get_matrix(inner_mc_id, ie.matrix_index)
            break

    return result


def render_champion_card(
    card_sc: SC5File,
    card_textures: list[Image.Image | None],
    portrait_sc: SC5File,
    portrait_textures: list[Image.Image | None],
    primary_form: str,
    secondary_form: str | None = None,
    portrait_scale: float = 0.55,
    card_export: str = "card_item_image_colored_champion",
) -> Image.Image | None:
    """Render a complete champion card with portrait and overlay.

    *primary_form*/*secondary_form*: frame labels like ``hero_unlocked``,
    ``evo_unlocked``.  Primary controls the notch, glow, and right-side
    diamond; secondary controls the left-side diamond.  If *secondary_form*
    is ``None``, the primary form is used everywhere.

    Returns a composited RGBA image or ``None`` on failure.
    """
    if secondary_form is None:
        secondary_form = primary_form

    # --- Render card overlay parts individually (preserving blend modes) ---
    original_find = card_sc._find_frame_by_label

    def _clean_glow_find(mc_id: int, label: str) -> int:
        mcd = card_sc.movie_clip_data.get(mc_id)
        if mcd is None:
            return 0
        # Champion cards use the hero_unlocked glow variant
        effective = "hero_unlocked" if label == "champion" and mc_id == 1001 else label
        for i, fl in enumerate(mcd.frame_labels):
            if fl == effective:
                return i
        # Glow sub-MCs: use clean frame (border only, no particle clouds)
        if mc_id in (990, 1000):
            return min(_GLOW_CLEAN_FRAME, len(mcd.frame_element_counts) - 1)
        for i, c in enumerate(mcd.frame_element_counts):
            if c > 0:
                return i
        return 0

    card_sc._find_frame_by_label = _clean_glow_find
    card_sc._render_children = _GLOW_BORDER_ONLY  # suppress sparkles

    card_obj = card_sc.exports.get(card_export)
    if card_obj is None:
        card_sc._find_frame_by_label = original_find
        del card_sc._render_children
        return None

    child_labels = {
        _CARD_CHILD_NOTCH_BASE: primary_form,
        _CARD_CHILD_GLOW: primary_form,
        _CARD_CHILD_DIAMOND_RIGHT: primary_form,
        _CARD_CHILD_DIAMOND_LEFT: secondary_form,
    }

    card_parts = card_sc._render_object(
        card_obj, card_textures, Matrix2x3.IDENTITY, set(),
        child_labels=child_labels,
    )
    card_sc._find_frame_by_label = original_find
    del card_sc._render_children

    if not card_parts:
        return None

    # --- Render portrait ---
    portrait_exports = list(portrait_sc.exports.values())
    if portrait_exports:
        portrait_obj = portrait_exports[0]
    elif portrait_sc.movie_clip_data:
        portrait_obj = next(iter(portrait_sc.movie_clip_data))
    else:
        return None

    portrait_parts = portrait_sc._render_object(
        portrait_obj, portrait_textures, Matrix2x3.IDENTITY, set(),
    )
    if not portrait_parts:
        return None

    portrait_result = _composite_parts(portrait_parts)
    if portrait_result is None:
        return None
    p_img, p_x, p_y = portrait_result

    # --- Scale and clip portrait to card mask ---
    # Shape 392 lives inside the glow MC hierarchy.  When rendering it in
    # isolation we must apply the same accumulated transform the hierarchy
    # would give it:  card MC → glow child → inner glow MC → shape 392.
    mask_matrix = _mask_hierarchy_transform(card_sc, card_obj, _clean_glow_find)
    mask_parts = card_sc._render_object(
        _CARD_PORTRAIT_MASK_SHAPE, card_textures, mask_matrix, set(),
    )
    if not mask_parts:
        return None
    mask_result = _composite_parts(mask_parts)
    if mask_result is None:
        return None
    m_img, m_x, m_y = mask_result
    m_alpha = np.array(m_img)[:, :, 3]
    m_binary = np.where(m_alpha > 0, 255, 0).astype(np.uint8)

    p_scaled = p_img.resize(
        (int(p_img.width * portrait_scale), int(p_img.height * portrait_scale)),
        Image.LANCZOS,
    )
    sp_x, sp_y = p_x * portrait_scale, p_y * portrait_scale

    # In the game the portrait is placed between the Masked / Unmasked
    # modifiers inside the inner glow MC (1000).  Its local origin sits
    # at MC 1000's (0,0), which is offset from the card root by the same
    # accumulated hierarchy transform that reaches Shape 392.  Apply that
    # transform so the portrait and mask share the same coordinate origin.
    sp_x += mask_matrix.tx
    sp_y += mask_matrix.ty

    clip_mask = Image.new("L", p_scaled.size, 0)
    clip_mask.paste(
        Image.fromarray(m_binary),
        (int(m_x - sp_x + 0.5), int(m_y - sp_y + 0.5)),
    )
    p_arr = np.array(p_scaled)
    cm_arr = np.array(clip_mask)
    p_arr[:, :, 3] = (
        p_arr[:, :, 3].astype(np.int32) * cm_arr.astype(np.int32) // 255
    ).astype(np.uint8)
    clipped_portrait = Image.fromarray(p_arr)

    # --- Composite everything onto a single canvas ---
    all_parts = [(clipped_portrait, sp_x, sp_y, 0)] + card_parts
    xmin = min(x for _, x, _, _ in all_parts) - 1
    ymin = min(y for _, _, y, _ in all_parts) - 1
    xmax = max(x + img.width for img, x, _, _ in all_parts) + 1
    ymax = max(y + img.height for img, _, y, _ in all_parts) + 1
    cw = int(xmax - xmin + 0.5)
    ch = int(ymax - ymin + 0.5)

    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))

    for img, xo, yo, blend in all_parts:
        px = int(xo - xmin + 0.5)
        py = int(yo - ymin + 0.5)
        if blend == 8:
            _additive_blend(canvas, img, px, py)
        else:
            canvas.alpha_composite(img, (px, py))

    return canvas
