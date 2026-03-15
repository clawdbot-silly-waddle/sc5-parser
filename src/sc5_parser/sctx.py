"""SCTX texture decoder for Supercell's streaming texture format."""

import struct
import zstandard
from PIL import Image

# Supercell pixel type codes found in SCTX streaming headers.
PIXEL_TYPE_BGRA = 70      # Raw BGRA8888
PIXEL_TYPE_ASTC_4x4 = 204 # ASTC 4×4 block compressed
PIXEL_TYPE_ASTC_8x8 = 212 # ASTC 8×8 block compressed


def decode_sctx(sctx_path: str) -> Image.Image:
    """Decode an SCTX texture file to a PIL RGBA Image.

    SCTX files contain a streaming header (FlatBuffer + metadata fields)
    followed by ZSTD-compressed texture data. The ``pixel_type`` field
    determines the pixel format:

    * 70  — raw BGRA8888
    * 204 — ASTC 4×4 block compressed
    * 212 — ASTC 8×8 block compressed
    """
    import texture2ddecoder

    with open(sctx_path, "rb") as f:
        data = f.read()

    off = 0
    streaming_len = struct.unpack("<I", data[off : off + 4])[0]
    off += 4
    streaming_data = data[off : off + streaming_len]
    off += streaming_len
    data_len = struct.unpack("<I", data[off : off + 4])[0]
    off += 4
    off += data_len

    sd_off = 0
    header_len = struct.unpack("<I", streaming_data[sd_off : sd_off + 4])[0]
    sd_off += 4
    sd_off += header_len
    pixel_type = struct.unpack("<I", streaming_data[sd_off : sd_off + 4])[0]
    sd_off += 4
    width = struct.unpack("<H", streaming_data[sd_off : sd_off + 2])[0]
    sd_off += 2
    height = struct.unpack("<H", streaming_data[sd_off : sd_off + 2])[0]

    compressed_tex = data[off:]
    dctx = zstandard.ZstdDecompressor()
    tex_data = dctx.decompress(
        compressed_tex, max_output_size=width * height * 4 * 2
    )

    if pixel_type == PIXEL_TYPE_BGRA:
        return Image.frombytes("RGBA", (width, height), tex_data, "raw", "BGRA")

    if pixel_type == PIXEL_TYPE_ASTC_4x4:
        decoded = texture2ddecoder.decode_astc(tex_data, width, height, 4, 4)
    elif pixel_type == PIXEL_TYPE_ASTC_8x8:
        decoded = texture2ddecoder.decode_astc(tex_data, width, height, 8, 8)
    else:
        raise ValueError(
            f"Unknown SCTX pixel type {pixel_type} in {sctx_path}"
        )

    return Image.frombytes("RGBA", (width, height), decoded, "raw", "BGRA")
