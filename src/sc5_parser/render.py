"""SC5 rendering: sprite extraction and scene-graph traversal.

All functions accept an ``SC5File`` instance as the first argument so
that they can be used as free functions without mutating parser state.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from sc5_parser.compositor import apply_color_transform, clip_to_mask, composite_parts
from sc5_parser.models import ColorTransform, Matrix2x3, RenderContext

if TYPE_CHECKING:
    from sc5_parser.parser import SC5File

# MovieClipModifier types (mirrored from parser for local use)
_MOD_MASK = 38
_MOD_MASKED = 39
_MOD_UNMASKED = 40

# Sentinel: movie clip has no frame element data
_NO_FRAME_ELEMENTS = 0xFFFFFFFF

# Maximum recursion depth for render_object
_MAX_RENDER_DEPTH = 50

# UV coordinate divisor (16-bit range → normalized)
_UV_DIVISOR = 65535.0


def _apply_blend(
    child_parts: list[tuple[Image.Image, float, float, int]],
    effective_blend: int,
) -> list[tuple[Image.Image, float, float, int]]:
    """Consolidate *child_parts* under a single blend mode if non-zero."""
    if effective_blend == 0 or not child_parts:
        return child_parts
    if len(child_parts) > 1:
        comp = composite_parts(
            [(im, x, y, 0) for im, x, y, _ in child_parts]
        )
        if comp:
            c_img, c_x, c_y = comp
            return [(c_img, c_x, c_y, effective_blend)]
        return []
    return [(im, x, y, effective_blend) for im, x, y, _ in child_parts]


def find_shapes_for_export(sc: SC5File, export_name: str) -> list[int]:
    """Return shape-list indices for every shape reachable from *export_name*."""
    mc_id = sc.exports.get(export_name)
    if mc_id is None:
        return []
    return collect_shapes(sc, mc_id, set())


def collect_shapes(sc: SC5File, obj_id: int, visited: set[int]) -> list[int]:
    """Recursively collect shape indices reachable from *obj_id*."""
    if obj_id in visited:
        return []
    visited.add(obj_id)
    indices = list(sc.shape_id_to_idx.get(obj_id, []))
    mcd = sc.movie_clip_data.get(obj_id)
    if mcd:
        for child_id in mcd.children_ids:
            indices.extend(collect_shapes(sc, child_id, visited))
    return indices


def render_shape(
    sc: SC5File,
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
    from sc5_parser.renderer import render_command

    shape = sc.shapes[shape_idx]
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
        tex = sc.textures[tex_idx]
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
    sc: SC5File,
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
    if depth > _MAX_RENDER_DEPTH:
        return []

    if _tex_arr_cache is None:
        _tex_arr_cache = {}

    rendered: list[tuple[Image.Image, float, float, int]] = []

    # If it's a shape, render it with the parent matrix baked into
    # vertex coordinates so the texture is sampled at full target
    # resolution (no lossy post-rasterisation upscale).
    # Pure translations skip the transform (just offset the result).
    shape_indices = sc.shape_id_to_idx.get(obj_id, [])
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
        img, x_off, y_off = render_shape(
            sc, si, texture_images, transform=mat_tuple,
            _tex_arr_cache=_tex_arr_cache,
        )
        if img is None:
            continue
        if color is not None:
            img = apply_color_transform(color, img)
        if _pure_xlate:
            x_off += parent_matrix.tx
            y_off += parent_matrix.ty
        rendered.append((img, x_off, y_off, blend_mode))

    # If it's a movie clip, process frame elements
    mcd = sc.movie_clip_data.get(obj_id)
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
            frame_idx = sc.find_frame_by_label(obj_id, frame_label)

        elements = sc.get_frame_elements(obj_id, frame_idx)

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
                mod_type = sc.modifiers.get(child_id)
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

                child_mat = sc.get_matrix(obj_id, elem.matrix_index)
                child_color = sc.get_color(obj_id, elem.color_index)
                combined = parent_matrix @ child_mat
                child_blend = (
                    mcd.children_blending[elem.child_index]
                    if elem.child_index < len(mcd.children_blending)
                    else 0
                )

                # Render child with normal compositing internally
                child_parts = render_object(
                    sc, child_id, texture_images, combined, child_visited,
                    child_color if child_color is not ColorTransform.IDENTITY else color,
                    effective_label, depth + 1,
                    blend_mode=0,
                    ctx=ctx,
                    _tex_arr_cache=_tex_arr_cache,
                )

                # If child has non-zero blend, composite fragments into one image
                # first, then apply the blend to the single result
                effective_blend = child_blend if child_blend != 0 else blend_mode
                child_parts = _apply_blend(child_parts, effective_blend)

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
        elif mcd.frame_elements_offset == _NO_FRAME_ELEMENTS:
            # No frame element data at all - render children with identity
            for idx, child_id in enumerate(mcd.children_ids):
                child_blend = (
                    mcd.children_blending[idx]
                    if idx < len(mcd.children_blending)
                    else 0
                )
                effective_blend = child_blend if child_blend != 0 else blend_mode
                child_parts = render_object(
                    sc, child_id, texture_images, parent_matrix, child_visited,
                    color, frame_label, depth + 1,
                    blend_mode=0,
                    ctx=ctx,
                    _tex_arr_cache=_tex_arr_cache,
                )
                child_parts = _apply_blend(child_parts, effective_blend)
                rendered.extend(child_parts)
        # else: selected frame explicitly has 0 elements - nothing visible

    return rendered


def extract_sprite(
    sc: SC5File,
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
    obj_id = sc.exports.get(export_name)
    if obj_id is None:
        return None

    rendered = render_object(
        sc, obj_id, texture_images, Matrix2x3.IDENTITY, set(),
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
    sc: SC5File,
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
    obj_id = sc.exports.get(export_name)
    if obj_id is None:
        return None

    rendered = render_object(
        sc, obj_id, texture_images, Matrix2x3.IDENTITY, set(),
        frame_label=frame_label,
        child_labels=child_labels,
        frame_index=frame_index,
        ctx=ctx,
    )

    if not rendered:
        return None

    return composite_parts(rendered)


def get_export_frame_info(sc: SC5File, export_name: str) -> dict[str, Any] | None:
    """Return frame count and labels for an export's root MC."""
    obj_id = sc.exports.get(export_name)
    if obj_id is None:
        return None
    mcd = sc.movie_clip_data.get(obj_id)
    if mcd is None:
        return None
    return {
        "frame_count": len(mcd.frame_element_counts),
        "frame_labels": mcd.frame_labels,
    }


def get_shape_bounds(sc: SC5File, shape_idx: int) -> dict[str, Any] | None:
    """Return the UV bounding box for a shape on its texture."""
    shape = sc.shapes[shape_idx]
    all_us: list[float] = []
    all_vs: list[float] = []
    tex_idx = None
    for cmd in shape["commands"]:
        tex_idx = cmd["texture_index"]
        if tex_idx >= len(sc.textures):
            continue
        tex = sc.textures[tex_idx]
        w, h = tex["width"], tex["height"]
        for _x, _y, u, v in cmd["vertices"]:
            all_us.append(u / _UV_DIVISOR * w)
            all_vs.append(v / _UV_DIVISOR * h)
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
