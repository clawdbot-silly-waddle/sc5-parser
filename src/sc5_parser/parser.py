"""SC v5 file parser.

Decodes the binary SC v5 container (header → FileDescriptor FlatBuffer →
ZSTD-compressed inner stream) and exposes shapes, movie-clips, textures
and named exports for downstream extraction.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np
import zstandard
from PIL import Image

from sc5_parser._schemas.sc.flash.SC2.DataStorage import DataStorage
from sc5_parser._schemas.sc.flash.SC2.ExportNames import ExportNames
from sc5_parser._schemas.sc.flash.SC2.FileDescriptor import FileDescriptor
from sc5_parser._schemas.sc.flash.SC2.MovieClips import MovieClips
from sc5_parser._schemas.sc.flash.SC2.Shapes import Shapes
from sc5_parser._schemas.sc.flash.SC2.Textures import Textures
from sc5_parser.renderer import render_command

ShapeDict = dict[str, Any]


class SC5File:
    """Parsed representation of an SC v5 file."""

    def __init__(self, sc_path: str | Path) -> None:
        self.sc_path = Path(sc_path)
        self.shapes: list[ShapeDict] = []
        self.exports: dict[str, int] = {}  # name → movie-clip / shape id
        self.textures: list[dict[str, Any]] = []
        self.movie_clips: dict[int, dict[str, Any]] = {}
        self.strings: list[str] = []
        self.vertices: list[tuple[float, float, int, int]] = []
        self._shape_id_to_idx: dict[int, list[int]] = {}
        self._parse()

    # ------------------------------------------------------------------
    def _parse(self) -> None:
        raw = self.sc_path.read_bytes()

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
            children = [
                {"id": clip.ChildrenIds(j)}
                for j in range(clip.ChildrenIdsLength())
            ]
            child_names = []
            for j in range(clip.ChildrenNameRefIdsLength()):
                ref = clip.ChildrenNameRefIds(j)
                if 0 < ref <= len(self.strings):
                    child_names.append(self.strings[ref - 1])
                else:
                    child_names.append("")
            self.movie_clips[mc_id] = {
                "id": mc_id,
                "children": children,
                "children_names": child_names,
                "frame_count": clip.FramesLength(),
            }
        pos += 4 + mc_size

        # MovieClipModifiers (skip)
        mod_size = struct.unpack("<I", inner[pos : pos + 4])[0]
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
    def extract_sprite(
        self,
        export_name: str,
        texture_images: list[Image.Image | None],
        output_path: str | Path | None = None,
    ) -> Image.Image | None:
        """Extract a named sprite, compositing all reachable shape commands."""
        shape_indices = self.find_shapes_for_export(export_name)
        if not shape_indices:
            return None

        rendered: list[tuple[Image.Image, float, float]] = []

        for si in shape_indices:
            shape = self.shapes[si]
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
                rendered.append((img, x_off, y_off))

        if not rendered:
            return None

        all_x_min = min(xo for _, xo, _ in rendered)
        all_y_min = min(yo for _, _, yo in rendered)
        all_x_max = max(xo + img.width for img, xo, _ in rendered)
        all_y_max = max(yo + img.height for _, _, yo in rendered)

        out_w = int(all_x_max - all_x_min + 0.5)
        out_h = int(all_y_max - all_y_min + 0.5)
        if out_w <= 0 or out_h <= 0 or out_w > 4096 or out_h > 4096:
            return None

        result = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
        for img, xo, yo in rendered:
            result.alpha_composite(img, (int(xo - all_x_min), int(yo - all_y_min)))

        if output_path:
            result.save(str(output_path))

        return result

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
