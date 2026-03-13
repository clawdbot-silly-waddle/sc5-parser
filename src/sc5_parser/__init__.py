"""SC v5 FlatBuffer sprite parser and extractor.

Parses Supercell's SC v5 format using the official FlatBuffer schemas
from sc-workshop/SupercellFlash and extracts named sprites from texture
atlases using polygon UV mapping.
"""

from sc5_parser.parser import SC5File
from sc5_parser.sctx import decode_sctx

__all__ = ["SC5File", "decode_sctx"]