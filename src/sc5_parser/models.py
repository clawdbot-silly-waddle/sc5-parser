"""Data models for SC v5 file structures.

Pure data classes with no rendering or image-processing dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


ShapeDict = dict[str, Any]


@dataclass
class Matrix2x3:
    """Affine 2×3 transformation matrix (a b c d tx ty)."""
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    tx: float = 0.0
    ty: float = 0.0

    IDENTITY: ClassVar[Matrix2x3]

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

    IDENTITY: ClassVar[ColorTransform]


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
    frame_finder: Any | None = None
    """Custom frame resolver.  Called as ``frame_finder(mc_id, label)``
    and should return a frame index.  When ``None`` the default
    :meth:`SC5File.find_frame_by_label` is used."""

    render_children: dict[int, frozenset[int]] | None = None
    """Per-MC child filter.  Keys are MC IDs; values are frozensets of
    allowed ``child_index`` values.  When a MC ID is present in this
    dict only the listed children are rendered."""

    inject_in_mask: Any | None = None
    """Content to inject into mask groups.  Keys are MC IDs; values are
    lists of ``(image, x, y, blend_mode)`` tuples that get composited
    into the mask group at the MASKED modifier boundary."""

    overlay_after_child: Any | None = None
    """Extra parts to insert after a specific child index during top-level
    MC rendering.  Keys are child indices; values are lists of
    ``(image, x, y, blend_mode)`` tuples.  Only consumed at depth 0 of
    the root ``render_object`` call (same scope as *child_labels*)."""
