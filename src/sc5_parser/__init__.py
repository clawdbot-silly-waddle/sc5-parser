"""SC v5 FlatBuffer sprite parser and extractor.

Parses Supercell's SC v5 format using the official FlatBuffer schemas
from sc-workshop/SupercellFlash and extracts named sprites from texture
atlases using polygon UV mapping.
"""

from sc5_parser.models import (
    ColorTransform,
    Matrix2x3,
    MovieClipData,
    RenderContext,
    ScalingGrid,
    ShapeDict,
    TextFieldData,
)
from sc5_parser.parser import (
    SC5File,
    apply_color_transform,
    blend_layer,
    clip_to_mask,
    composite_parts,
)
from sc5_parser.sctx import decode_pixel_data, decode_sctx

__all__ = [
    "ColorTransform",
    "Matrix2x3",
    "MovieClipData",
    "RenderContext",
    "SC5File",
    "ScalingGrid",
    "ShapeDict",
    "TextFieldData",
    "apply_color_transform",
    "blend_layer",
    "clip_to_mask",
    "composite_parts",
    "decode_pixel_data",
    "decode_sctx",
]