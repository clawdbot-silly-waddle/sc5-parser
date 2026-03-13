# sc5-parser

Parser and sprite extractor for **Supercell's SC v5** FlatBuffer format, used
in games like Clash Royale, Brawl Stars and Clash of Clans.

Extracts individual named sprites from texture atlases using proper polygon
UV mapping (triangle-strip rasterisation), not bounding-box crops.

## Installation

```bash
uv pip install .
# or
pip install .
```

## Usage

```bash
# Show file structure info
sc5-parser ui_card_items.sc --info

# List all named exports
sc5-parser ui_card_items.sc --list

# Extract all sprites (SCTX files must be in same directory)
sc5-parser ui_card_items.sc -o sprites/

# Extract specific sprites
sc5-parser ui_card_items.sc --extract icon_crystal card_item_frame -o sprites/

# Specify a separate directory for SCTX texture files
sc5-parser ui_card_items.sc --sctx-dir /path/to/textures -o sprites/
```

## SC v5 File Format

```
┌─────────────────────────────────────────────────┐
│  'SC' (2 bytes)                                 │
│  version (u32 LE = 5)                           │
│  fd_size (u32)                                  │
│  FileDescriptor FlatBuffer (fd_size bytes)       │
│  ZSTD-compressed inner stream                    │
│  ├── DataStorage FlatBuffer (size-prefixed)      │
│  └── Chunks at resources_offset:                 │
│      ├── ExportNames                             │
│      ├── TextFields                              │
│      ├── Shapes                                  │
│      ├── MovieClips                              │
│      ├── MovieClipModifiers                      │
│      └── Textures                                │
└─────────────────────────────────────────────────┘
```

The FileDescriptor sits **before** the ZSTD-compressed payload, not inside it.
Textures are stored externally in `.sctx` files (ZSTD-compressed ASTC 8×8).

Shape draw commands reference vertices from the DataStorage's bitmap-points
buffer. Each vertex is 12 bytes: `x(f32) + y(f32) + u(u16) + v(u16)`, with UV
coordinates normalised to 0–65535.

## FlatBuffer Schemas

The `_schemas/` directory contains Python code generated from the official
FlatBuffer schemas in
[sc-workshop/SupercellFlash](https://github.com/sc-workshop/SupercellFlash)
(`supercell-flash/sc2_schemas/`).

To regenerate:

```bash
flatc --python *.fbs
```

## License

MIT
