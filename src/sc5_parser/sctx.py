"""SCTX texture decoder for Supercell's streaming texture format."""

import struct
import zstandard
from PIL import Image


def decode_sctx(sctx_path: str) -> Image.Image:
    """Decode an SCTX texture file to a PIL RGBA Image.

    SCTX files contain ZSTD-compressed ASTC 8×8 texture data with a
    streaming header that stores pixel type and dimensions.
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
    _pixel_type = struct.unpack("<I", streaming_data[sd_off : sd_off + 4])[0]
    sd_off += 4
    width = struct.unpack("<H", streaming_data[sd_off : sd_off + 2])[0]
    sd_off += 2
    height = struct.unpack("<H", streaming_data[sd_off : sd_off + 2])[0]

    compressed_tex = data[off:]
    dctx = zstandard.ZstdDecompressor()
    tex_data = dctx.decompress(
        compressed_tex, max_output_size=width * height * 4 * 2
    )

    decoded = texture2ddecoder.decode_astc(tex_data, width, height, 8, 8)
    return Image.frombytes("RGBA", (width, height), decoded, "raw", "BGRA")
