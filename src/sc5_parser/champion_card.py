"""Champion card compositor for Clash Royale.

Renders a complete champion card image by compositing a portrait into the
card frame overlay from ``ui_card_items.sc``.  This module is a *consumer*
of the ``sc5_parser`` library - it uses only the public API.

Champion Card Structure (MC 1008 ``card_item_image_colored_champion``)
----------------------------------------------------------------------

::

    child[0] MC 982  "hero_activate_anim"  - full-width base notch (99px)
    child[1] MC 987  "bg_full"             - right half overlay (50px)
    child[2] MC 988  "bg_right"            - left half overlay (50px)
    child[3] MC 1001 "evo_glow"            - glow effect
    child[4] MC 1005 "bg_left"             - CENTER diamond slot
    child[5] MC 1005 "diamond_center"      - RIGHT diamond slot
    child[6] MC 1005 "diamond_right"       - LEFT diamond slot
    child[7] MC 572  "frame_anim"          - animation (empty)

**NOTE**: Instance names don't match positions - "bg_left" is actually the
center diamond, "diamond_center" is the right one, etc.

Rendering rules
~~~~~~~~~~~~~~~

- **Single form** (hero-only or evo-only): Show base (child 0) + center
  diamond (child 4).  Hide halves (children 1, 2) and outer diamonds (5, 6).
- **Dual form** (hero+evo): Show base (child 0) + both halves (children 1, 2)
  + outer diamonds (children 5, 6).  The base fills the seam between halves.
- **Diamond labels**: ``evo_locked``/``evo_unlocked``/``evo_active`` (purple/gold),
  ``hero_locked``/``hero_unlocked`` (gold).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PIL import Image

from sc5_parser.compositor import composite_parts
from sc5_parser.parser import (
    Matrix2x3,
    RenderContext,
    SC5File,
)


@dataclass
class CardRenderResult:
    """Result of ``render_champion_card`` including compositing metadata."""
    image: Image.Image
    bounds: tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)

# -- MC 1008 child indices -------------------------------------------------

CHILD_NOTCH_BASE = 0      # MC 982: full-width notch
CHILD_NOTCH_RIGHT = 1     # MC 987: right half overlay (unused; for reference)
CHILD_NOTCH_LEFT = 2      # MC 988: left half overlay (unused; for reference)
CHILD_GLOW = 3            # MC 1001: frame glow border + mask
CHILD_DIAMOND_CENTER = 4  # MC 1005 (actually "bg_left")
CHILD_DIAMOND_RIGHT = 5   # MC 1005 @ tx=18.5
CHILD_DIAMOND_LEFT = 6    # MC 1005 @ tx=-18.6

# -- Glow rendering config -------------------------------------------------

# Glow sub-MCs (990=evo, 1000=hero) have 34 frames.  Frame 29 is a clean
# border with the outline shape, clipping mask, and overlay children but
# no particle/sparkle effects.
_GLOW_CLEAN_FRAME = 29

# Shape 392 is the card portrait clipping mask - a solid rounded rectangle
# matching the interior of the champion frame border.
_PORTRAIT_MASK_SHAPE = 392

# MC 973 is the diagonal shimmer sweep — it extends well beyond the card
# bounds and causes visible bouncing when animated.  Excluded from
# animation by default.
_SHIMMER_MC = 973

# The game composites a frame border on top of the card base per
# ``card_forms.toml``.  MC 974 (``card_item_frame``) provides the rounded
# golden border that matches the in-game reference for champion cards.
_FRAME_EXPORT = "card_item_frame"


# -- Internal helpers -------------------------------------------------------

def _make_frame_finder(
    sc: SC5File,
    animation_frame: int | None = None,
    animation_exclude: set[int] | None = None,
    animation_only: set[int] | None = None,
) -> Callable[[int, str], int]:
    """Build a frame-finder callback for champion card rendering.

    The returned callable resolves frame labels with special handling for
    glow sub-MCs (selecting the clean-border frame) and the champion →
    hero_unlocked alias.

    When *animation_frame* is set, pure-animation MCs (many frames, no
    meaningful labels) cycle through their frames.  MCs with labelled
    frames (state variants) are left alone.

    *animation_exclude*: MC IDs to skip even if they match the animation
    heuristic.

    *animation_only*: when set, ONLY these MC IDs animate (overrides the
    auto-detection heuristic).
    """
    # Pre-compute which MCs are pure animations.
    _anim_ids: set[int] = set()
    if animation_frame is not None:
        if animation_only is not None:
            _anim_ids = animation_only
        else:
            exclude = animation_exclude or set()
            for mid, mcd in sc.movie_clip_data.items():
                if (mid not in exclude
                        and len(mcd.frame_element_counts) > 10
                        and not any(mcd.frame_labels)):
                    _anim_ids.add(mid)

    def _find(mc_id: int, label: str) -> int:
        mcd = sc.movie_clip_data.get(mc_id)
        if mcd is None:
            return 0
        n_frames = len(mcd.frame_element_counts)

        # Animation mode: cycle pure-animation MCs (no labelled frames).
        if mc_id in _anim_ids:
            return animation_frame % n_frames  # type: ignore[operator]

        effective = (
            "hero_unlocked" if label == "champion" and mc_id == 1001
            else label
        )
        for i, fl in enumerate(mcd.frame_labels):
            if fl == effective:
                return i
        # Glow sub-MCs: use clean frame (border only, no particle clouds)
        if mc_id in (990, 1000):
            return min(_GLOW_CLEAN_FRAME, n_frames - 1)
        for i, c in enumerate(mcd.frame_element_counts):
            if c > 0:
                return i
        return 0

    return _find


def _mask_hierarchy_transform(
    card_sc: SC5File,
    card_mc_id: int,
    frame_finder: Callable[[int, str], int],
    glow_label: str = "",
) -> tuple[Matrix2x3, int | None]:
    """Compute the accumulated transform from the card MC to the mask shape.

    Traces: card MC → child[CHILD_GLOW] (MC 1001) → inner glow MC
    (MC 990/1000) → Shape 392, multiplying matrices along the path.

    Returns ``(transform, inner_glow_mc_id)`` where *inner_glow_mc_id* is
    the MC that contains the mask group (Mask / Masked / Unmasked).
    """
    result = Matrix2x3.IDENTITY
    card_mcd = card_sc.movie_clip_data.get(card_mc_id)
    if card_mcd is None:
        return result, None

    # Step 1: card MC → glow child
    card_fe = card_sc.get_frame_elements(card_mc_id, 0)
    glow_mc_id: int | None = None
    for elem in card_fe:
        if elem.child_index == CHILD_GLOW:
            if CHILD_GLOW >= len(card_mcd.children_ids):
                return result, None
            result = result @ card_sc.get_matrix(card_mc_id, elem.matrix_index)
            glow_mc_id = card_mcd.children_ids[CHILD_GLOW]
            break
    if glow_mc_id is None:
        return result, None

    # Step 2: glow MC → inner glow MC (990 or 1000)
    glow_mcd = card_sc.movie_clip_data.get(glow_mc_id)
    if glow_mcd is None:
        return result, None
    glow_frame = frame_finder(glow_mc_id, glow_label)
    glow_fe = card_sc.get_frame_elements(glow_mc_id, glow_frame)
    if not glow_fe:
        return result, None
    if glow_fe[0].child_index >= len(glow_mcd.children_ids):
        return result, None
    result = result @ card_sc.get_matrix(glow_mc_id, glow_fe[0].matrix_index)
    inner_mc_id = glow_mcd.children_ids[glow_fe[0].child_index]

    # Step 3: inner glow MC → portrait mask shape
    inner_mcd = card_sc.movie_clip_data.get(inner_mc_id)
    if inner_mcd is None:
        return result, inner_mc_id
    inner_frame = min(_GLOW_CLEAN_FRAME, len(inner_mcd.frame_element_counts) - 1)
    inner_fe = card_sc.get_frame_elements(inner_mc_id, inner_frame)
    for ie in inner_fe:
        if (ie.child_index < len(inner_mcd.children_ids)
                and inner_mcd.children_ids[ie.child_index]
                == _PORTRAIT_MASK_SHAPE):
            result = result @ card_sc.get_matrix(inner_mc_id, ie.matrix_index)
            break

    return result, inner_mc_id


# -- Public API -------------------------------------------------------------

def render_champion_card(
    card_sc: SC5File,
    card_textures: list[Image.Image | None],
    portrait_sc: SC5File,
    portrait_textures: list[Image.Image | None],
    primary_form: str = "hero_unlocked",
    secondary_form: str = "evo_unlocked",
    portrait_scale: float = 0.55,
    card_export: str = "card_item_image_colored_champion",
    render_scale: float = 1.0,
    animation_frame: int | None = None,
    canvas_bounds: tuple[float, float, float, float] | None = None,
) -> CardRenderResult | None:
    """Render a complete champion card with portrait and overlay.

    *primary_form*/*secondary_form*: frame labels like ``hero_unlocked``,
    ``evo_unlocked``.  Primary controls the notch, glow, and right-side
    diamond; secondary controls the left-side diamond.  Champion cards are
    inherently dual-form, so both labels default to produce a dual-form
    champion with gold (hero_unlocked) and purple (evo_unlocked) diamonds.

    Pass ``secondary_form=primary_form`` to force single-form (center
    diamond only).

    *render_scale*: multiplier for the output resolution.  At 1.0 shapes
    are rasterised at native SC coordinates (~130 px wide); higher values
    produce proportionally larger output with texture sampling at the
    target density, avoiding nearest-neighbour pixelation.

    *animation_frame*: when set, pure-animation MCs (shimmer, glow,
    diamond glow) cycle through their animation frames.  Render 0..N-1
    and combine into APNG for animated output.

    *canvas_bounds*: optional ``(xmin, ymin, xmax, ymax)`` to lock the
    compositing bounding box.  Use this for animation sequences to
    prevent jitter from per-frame bounding-box variation.

    Returns a `CardRenderResult` with the composited RGBA image and
    bounding box, or ``None`` on failure.
    """
    if render_scale <= 0:
        raise ValueError("render_scale must be positive")

    card_obj = card_sc.exports.get(card_export)
    if card_obj is None:
        return None

    frame_finder = _make_frame_finder(
        card_sc,
        animation_frame=animation_frame,
        animation_only={_SHIMMER_MC},
    )

    # Base matrix: identity scaled by render_scale for higher-res output.
    base_matrix = (
        Matrix2x3(a=render_scale, b=0, c=0, d=render_scale, tx=0, ty=0)
        if render_scale != 1.0
        else Matrix2x3.IDENTITY
    )

    # --- Render portrait ---------------------------------------------------
    portrait_exports = list(portrait_sc.exports.values())
    if portrait_exports:
        portrait_obj = portrait_exports[0]
    elif portrait_sc.movie_clip_data:
        portrait_obj = next(iter(portrait_sc.movie_clip_data))
    else:
        return None

    portrait_parts = portrait_sc.render_object(
        portrait_obj, portrait_textures, Matrix2x3.IDENTITY, set(),
    )
    if not portrait_parts:
        return None

    portrait_result = composite_parts(portrait_parts)
    if portrait_result is None:
        return None
    p_img, p_x, p_y = portrait_result

    # Scale portrait to fit the card at render_scale resolution.
    effective_portrait_scale = portrait_scale * render_scale
    p_scaled = p_img.resize(
        (int(p_img.width * effective_portrait_scale),
         int(p_img.height * effective_portrait_scale)),
        Image.LANCZOS,
    )

    # Compute the accumulated hierarchy transform so we can place the
    # portrait in card-root coordinates for mask clipping.
    mask_matrix, inner_mc_id = _mask_hierarchy_transform(
        card_sc, card_obj, frame_finder, glow_label=primary_form,
    )
    # Portrait position in scaled card-root coordinates.
    sp_x = p_x * effective_portrait_scale + mask_matrix.tx * render_scale
    sp_y = p_y * effective_portrait_scale + mask_matrix.ty * render_scale

    # --- Frame overlay (rendered separately, injected between glow & diamonds)
    frame_parts: list[tuple[Image.Image, float, float, int]] = []
    frame_mc_id = card_sc.exports.get(_FRAME_EXPORT)
    if frame_mc_id is not None:
        frame_parts = card_sc.render_object(
            frame_mc_id, card_textures, base_matrix, set(),
        )

    # Build render context: custom frame finder, portrait injection into
    # the mask group, and frame overlay after the glow child.
    ctx = RenderContext(
        frame_finder=frame_finder,
        inject_in_mask=(
            {inner_mc_id: [(p_scaled, sp_x, sp_y, 0)]}
            if inner_mc_id is not None
            else None
        ),
        overlay_after_child={CHILD_GLOW: frame_parts} if frame_parts else None,
    )

    # --- Render card (single pass) -----------------------------------------
    # MC 1008's natural child order determines z-order:
    #   notch (0) -> glow+portrait (3) -> [frame overlay] -> diamonds (4-6)
    # Only the base notch is shown (not halves); diamonds use form labels.
    child_labels: dict[int, str] = {
        CHILD_NOTCH_BASE: primary_form,
        CHILD_GLOW: primary_form,
    }
    dual = secondary_form != primary_form
    if dual:
        child_labels[CHILD_DIAMOND_RIGHT] = primary_form
        child_labels[CHILD_DIAMOND_LEFT] = secondary_form
    else:
        child_labels[CHILD_DIAMOND_CENTER] = primary_form

    card_parts = card_sc.render_object(
        card_obj, card_textures, base_matrix, set(),
        child_labels=child_labels,
        ctx=ctx,
    )
    if not card_parts:
        return None

    # --- Composite ---------------------------------------------------------
    # When no explicit bounds are given, add 1px padding for anti-alias bleed.
    if canvas_bounds is None:
        pad_bounds: tuple[float, float, float, float] | None = (
            min(x for _, x, _, _ in card_parts) - 1,
            min(y for _, _, y, _ in card_parts) - 1,
            max(x + img.width for img, x, _, _ in card_parts) + 1,
            max(y + img.height for img, _, y, _ in card_parts) + 1,
        )
    else:
        pad_bounds = canvas_bounds

    result = composite_parts(card_parts, bounds=pad_bounds)
    if result is None:
        return None
    image, ox, oy = result
    return CardRenderResult(
        image=image,
        bounds=(ox, oy, ox + image.width, oy + image.height),
    )
