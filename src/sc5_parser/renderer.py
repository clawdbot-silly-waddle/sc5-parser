"""Triangle-strip rasterizer for SC shape commands.

Each shape command defines a triangle strip with interleaved XY (local-space)
and UV (texture-space, normalised 0–65535) coordinates.  The renderer
rasterises each triangle using scanline barycentric interpolation and
samples the source texture atlas.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

MAX_SPRITE_DIM = 4096


def triangulate_strip(n: int) -> list[tuple[int, int, int]]:
    """Return triangle index triples from *n* triangle-strip vertices."""
    tris: list[tuple[int, int, int]] = []
    for i in range(n - 2):
        if i % 2 == 0:
            tris.append((i, i + 1, i + 2))
        else:
            tris.append((i, i + 2, i + 1))
    return tris


def render_command(
    cmd_verts: list[tuple[float, float, int, int]],
    tex_img: Image.Image,
    tex_w: int,
    tex_h: int,
) -> tuple[Image.Image | None, float, float]:
    """Rasterise a shape command to an RGBA image.

    Parameters
    ----------
    cmd_verts:
        List of ``(x, y, u_raw, v_raw)`` tuples.  *u_raw*/*v_raw* are
        normalised 0–65535 and mapped to pixel coordinates internally.
    tex_img:
        Decoded RGBA texture atlas.
    tex_w, tex_h:
        Texture dimensions in pixels.

    Returns
    -------
    ``(image, x_offset, y_offset)`` — the rendered sprite fragment and
    its position relative to the shape's local origin.  Returns
    ``(None, 0, 0)`` when rendering is impossible.
    """
    if len(cmd_verts) < 3:
        return None, 0, 0

    xs = [v[0] for v in cmd_verts]
    ys = [v[1] for v in cmd_verts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    out_w = max(1, int(x_max - x_min + 1.5))
    out_h = max(1, int(y_max - y_min + 1.5))

    if out_w > MAX_SPRITE_DIM or out_h > MAX_SPRITE_DIM:
        return None, 0, 0

    tex_arr = np.array(tex_img)
    out_arr = np.zeros((out_h, out_w, 4), dtype=np.uint8)

    for i0, i1, i2 in triangulate_strip(len(cmd_verts)):
        x0, y0, u0_raw, v0_raw = cmd_verts[i0]
        x1, y1, u1_raw, v1_raw = cmd_verts[i1]
        x2, y2, u2_raw, v2_raw = cmd_verts[i2]

        tu0 = u0_raw / 65535.0 * tex_w
        tv0 = v0_raw / 65535.0 * tex_h
        tu1 = u1_raw / 65535.0 * tex_w
        tv1 = v1_raw / 65535.0 * tex_h
        tu2 = u2_raw / 65535.0 * tex_w
        tv2 = v2_raw / 65535.0 * tex_h

        ox0 = x0 - x_min
        oy0 = y0 - y_min
        ox1 = x1 - x_min
        oy1 = y1 - y_min
        ox2 = x2 - x_min
        oy2 = y2 - y_min

        denom = (oy1 - oy2) * (ox0 - ox2) + (ox2 - ox1) * (oy0 - oy2)
        if abs(denom) < 1e-6:
            continue
        inv_denom = 1.0 / denom

        tri_y_min = max(0, int(min(oy0, oy1, oy2)))
        tri_y_max = min(out_h - 1, int(max(oy0, oy1, oy2) + 0.5))

        for py in range(tri_y_min, tri_y_max + 1):
            intersections: list[float] = []
            edges = [
                (ox0, oy0, ox1, oy1),
                (ox1, oy1, ox2, oy2),
                (ox2, oy2, ox0, oy0),
            ]
            for ex0, ey0, ex1, ey1 in edges:
                if ey0 == ey1:
                    continue
                if (ey0 <= py < ey1) or (ey1 <= py < ey0):
                    t = (py - ey0) / (ey1 - ey0)
                    intersections.append(ex0 + t * (ex1 - ex0))

            if len(intersections) < 2:
                continue

            px_min = max(0, int(min(intersections)))
            px_max = min(out_w - 1, int(max(intersections) + 0.5))

            for px in range(px_min, px_max + 1):
                w0 = (
                    (oy1 - oy2) * (px - ox2) + (ox2 - ox1) * (py - oy2)
                ) * inv_denom
                w1 = (
                    (oy2 - oy0) * (px - ox2) + (ox0 - ox2) * (py - oy2)
                ) * inv_denom
                w2 = 1.0 - w0 - w1

                if w0 < -0.01 or w1 < -0.01 or w2 < -0.01:
                    continue

                su = w0 * tu0 + w1 * tu1 + w2 * tu2
                sv = w0 * tv0 + w1 * tv1 + w2 * tv2

                tx = max(0, min(tex_w - 1, int(su + 0.5)))
                ty = max(0, min(tex_h - 1, int(sv + 0.5)))

                out_arr[py, px] = tex_arr[ty, tx]

    result = Image.fromarray(out_arr, "RGBA")
    return result, x_min, y_min
