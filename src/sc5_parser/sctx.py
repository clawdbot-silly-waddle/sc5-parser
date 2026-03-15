"""SCTX texture decoder for Supercell's streaming texture format.

Also provides :func:`decode_pixel_data` for decoding raw texture buffers
in any of the SC5 pixel formats (RGBA8, RGBA4, RGB565, LA8, L8).
"""

import struct
import zstandard
import numpy as np
from PIL import Image

# Supercell pixel type codes found in SCTX streaming headers.
PIXEL_TYPE_BGRA = 70      # Raw BGRA8888
PIXEL_TYPE_ASTC_4x4 = 204 # ASTC 4×4 block compressed
PIXEL_TYPE_ASTC_8x8 = 212 # ASTC 8×8 block compressed

# SC5 embedded pixel format codes (from SWFTexture.h PixelFormat enum).
SC5_RGBA8 = 0
SC5_RGBA4 = 2
SC5_RGB5_A1 = 3
SC5_RGB565 = 4
SC5_LA8 = 6    # Luminance8 + Alpha8
SC5_L8 = 10   # Luminance8


def decode_pixel_data(
    data: bytes, width: int, height: int, pixel_format: int,
) -> Image.Image:
    """Decode a raw pixel buffer to an RGBA PIL Image.

    *pixel_format* uses the SC5 embedded codes (0/2/3/4/6/10).
    """
    if pixel_format == SC5_RGBA8:
        return Image.frombytes("RGBA", (width, height), data)

    if pixel_format == SC5_RGBA4:
        arr = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
        r = ((arr >> 12) & 0xF) * 17
        g = ((arr >> 8) & 0xF) * 17
        b = ((arr >> 4) & 0xF) * 17
        a = (arr & 0xF) * 17
        out = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
        return Image.fromarray(out, "RGBA")

    if pixel_format == SC5_RGB5_A1:
        arr = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
        r = ((arr >> 11) & 0x1F) * 255 // 31
        g = ((arr >> 6) & 0x1F) * 255 // 31
        b = ((arr >> 1) & 0x1F) * 255 // 31
        a = (arr & 1) * 255
        out = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
        return Image.fromarray(out, "RGBA")

    if pixel_format == SC5_RGB565:
        arr = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
        r = ((arr >> 11) & 0x1F) * 255 // 31
        g = ((arr >> 5) & 0x3F) * 255 // 63
        b = (arr & 0x1F) * 255 // 31
        a = np.full_like(r, 255)
        out = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
        return Image.fromarray(out, "RGBA")

    if pixel_format == SC5_LA8:
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 2)
        l = arr[:, :, 0]
        a = arr[:, :, 1]
        out = np.stack([l, l, l, a], axis=-1)
        return Image.fromarray(out, "RGBA")

    if pixel_format == SC5_L8:
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
        out = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)
        return Image.fromarray(out, "RGBA")

    raise ValueError(f"Unknown SC5 pixel format {pixel_format}")


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
