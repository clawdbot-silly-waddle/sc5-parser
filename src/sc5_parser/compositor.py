"""Image compositing utilities for SC5 rendering.

Provides blend-layer operations, alpha masking, and multi-part
compositing with support for all 14 Flash blend modes.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from sc5_parser.models import ColorTransform


def apply_color_transform(ct: ColorTransform, img: Image.Image) -> Image.Image:
    """Apply a color transform to an RGBA image."""
    if ct is ColorTransform.IDENTITY:
        return img
    if (ct.r_mul == 255 and ct.g_mul == 255 and ct.b_mul == 255
            and ct.alpha == 255
            and ct.r_add == 0 and ct.g_add == 0 and ct.b_add == 0):
        return img
    arr = np.array(img, dtype=np.int32)
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
    arr[:, :, 0] = np.clip(r * ct.r_mul // 255 + ct.r_add, 0, 255)
    arr[:, :, 1] = np.clip(g * ct.g_mul // 255 + ct.g_add, 0, 255)
    arr[:, :, 2] = np.clip(b * ct.b_mul // 255 + ct.b_add, 0, 255)
    arr[:, :, 3] = np.clip(a * ct.alpha // 255, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


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
