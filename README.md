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

### Frame Labels

MovieClips can have named frames (state selectors). Use `--frame-label` to
select a frame by name for all child MCs recursively:

```bash
# All children use "hero_unlocked" frame
sc5-parser ui_card_items.sc --extract card_item_image_colored_champion \
  --frame-label hero_unlocked -o sprites/
```

### Per-Child Frame Labels

For multi-form sprites (e.g. champion cards with both evo and hero states),
use `--child-labels` to assign individual frame labels to direct children of
the top-level MC. Only children listed are rendered; others are hidden.

```bash
# Champion card with hero base + evo left half + center diamond
sc5-parser ui_card_items.sc --extract card_item_image_colored_champion \
  --child-labels "0:hero_unlocked,4:evo_unlocked" -o sprites/
```

Format: `INDEX:LABEL,INDEX:LABEL,...` where INDEX is the child position in
the MC's children array.

#### Champion Card Structure (MC 1008)

```
child[0] MC 982  "hero_activate_anim"  — full-width base notch (99px)
child[1] MC 987  "bg_full"             — right half overlay (50px)
child[2] MC 988  "bg_right"            — left half overlay (50px)
child[3] MC 1001 "evo_glow"            — glow effect
child[4] MC 1005 "bg_left"             — CENTER diamond slot
child[5] MC 1005 "diamond_center"      — RIGHT diamond slot
child[6] MC 1005 "diamond_right"       — LEFT diamond slot
child[7] MC 572  "frame_anim"          — animation (empty)
```

**NOTE**: Instance names don't match positions — "bg_left" is actually the
center diamond, "diamond_center" is the right one, etc.

**Rendering rules:**
- **Single form** (hero-only or evo-only): Show base (child 0) + center
  diamond (child 4). Hide halves (children 1,2) and outer diamonds (5,6).
- **Dual form** (hero+evo): Show base (child 0) + both halves (children 1,2)
  + outer diamonds (children 5,6). The base fills the seam between halves.
- **Diamond labels**: `evo_locked`/`evo_unlocked`/`evo_active` (purple/gold),
  `hero_locked`/`hero_unlocked` (gold). The frame colour difference between
  evo and hero diamonds is baked into the sprite art (shapes 491 vs 493).

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

### Known Gotchas

- **Frame elements vector**: The FlatBuffer schema declares
  `movieclips_frame_elements: [ushort]` but flatc generates Python code that
  reads `[ubyte]`. The parser works around this by reading raw bytes at the
  vtable offset as `uint16`.
- **ColorTransform math**: Intermediate values can reach 255×255 = 65025,
  which overflows `int16`. All arithmetic uses `int32`.
- **Blend modes**: Mode 0 = normal alpha composite, mode 8 = additive. Other
  modes exist but are rarely used. Additive blending requires compositing
  child fragments into an intermediate image first.
- **Masking**: MovieClipModifiers (types 38/39/40) implement a mask state
  machine — the mask child's alpha clips subsequent masked children.

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
