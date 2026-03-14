"""Command-line interface for sc5-parser."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sc5_parser.parser import SC5File
from sc5_parser.sctx import decode_sctx


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="sc5-parser",
        description="Parse and extract sprites from Supercell SC v5 files.",
    )
    ap.add_argument("sc_file", help="Path to an .sc file")
    ap.add_argument(
        "--sctx-dir",
        help="Directory containing .sctx texture files "
        "(default: same directory as sc_file)",
    )
    ap.add_argument(
        "-o",
        "--output-dir",
        help="Output directory for extracted PNGs "
        "(default: ./extracted/ next to sc_file)",
    )
    ap.add_argument("--list", action="store_true", help="List all exports")
    ap.add_argument(
        "--extract",
        nargs="*",
        metavar="NAME",
        help="Export names to extract (omit for all)",
    )
    ap.add_argument(
        "--frame-label",
        metavar="LABEL",
        help="Frame label to select for child MCs "
        "(e.g. evo_unlocked, hero_unlocked)",
    )
    ap.add_argument(
        "--child-labels",
        metavar="SPEC",
        help="Per-child frame labels for the top-level MC. Format: "
        '"INDEX:LABEL,INDEX:LABEL,..." e.g. "0:hero_unlocked,4:evo_unlocked". '
        "Only listed children are rendered; others are hidden.",
    )
    ap.add_argument(
        "--info", action="store_true", help="Show file structure summary"
    )

    args = ap.parse_args(argv)
    sc = SC5File(args.sc_file)

    if args.info:
        _print_info(sc)
        return

    if args.list:
        _print_exports(sc)
        return

    if args.extract is not None or args.output_dir:
        _extract(sc, args)
        return

    # Default: show info
    _print_info(sc)


# ------------------------------------------------------------------
def _print_info(sc: SC5File) -> None:
    print(f"File:        {sc.sc_path}")
    print(f"Shapes:      {len(sc.shapes)}")
    print(f"Movie clips: {len(sc.movie_clips)}")
    print(f"Textures:    {len(sc.textures)}")
    print(f"Exports:     {len(sc.exports)}")
    print(f"Vertices:    {len(sc.vertices)}")
    for i, tex in enumerate(sc.textures):
        print(
            f"  Texture {i}: {tex['width']}×{tex['height']}  "
            f"pixel_type={tex['pixel_type']}  file={tex['external']}"
        )


def _print_exports(sc: SC5File) -> None:
    for name in sorted(sc.exports):
        mc_id = sc.exports[name]
        shapes = sc.find_shapes_for_export(name)
        regions: list[str] = []
        for si in shapes:
            b = sc.get_shape_bounds(si)
            if b and b["width"] > 0:
                regions.append(f"{b['width']:.0f}×{b['height']:.0f}")
        print(
            f"  {name}  (id={mc_id}, shapes={len(shapes)}, "
            f"regions={', '.join(regions) or '—'})"
        )


def _extract(sc: SC5File, args: argparse.Namespace) -> None:
    sctx_dir = args.sctx_dir or str(sc.sc_path.parent)
    output_dir = args.output_dir or os.path.join(
        str(sc.sc_path.parent), "extracted"
    )
    os.makedirs(output_dir, exist_ok=True)

    texture_images = _load_textures(sc, sctx_dir)

    child_labels = _parse_child_labels(args.child_labels)

    names = args.extract if args.extract else list(sc.exports.keys())

    extracted = 0
    for name in names:
        out_path = os.path.join(output_dir, f"{name}.png")
        result = sc.extract_sprite(
            name, texture_images, out_path,
            frame_label=args.frame_label,
            child_labels=child_labels,
        )
        if result:
            print(f"  Extracted: {name} ({result.width}×{result.height})")
            extracted += 1
        else:
            print(f"  Skip: {name} (no visible shapes)")

    print(f"\nExtracted {extracted}/{len(names)} sprites to {output_dir}")


def _parse_child_labels(spec: str | None) -> dict[int, str] | None:
    """Parse ``"0:label_a,4:label_b"`` into ``{0: 'label_a', 4: 'label_b'}``."""
    if not spec:
        return None
    result: dict[int, str] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if ":" not in pair:
            print(
                f"WARNING: bad child-label pair '{pair}' (expected INDEX:LABEL)",
                file=sys.stderr,
            )
            continue
        idx_str, label = pair.split(":", 1)
        try:
            result[int(idx_str)] = label
        except ValueError:
            print(
                f"WARNING: bad child index '{idx_str}' in child-labels",
                file=sys.stderr,
            )
    return result or None


def _load_textures(sc: SC5File, sctx_dir: str) -> list:
    images: list = []
    for tex in sc.textures:
        if tex["external"]:
            sctx_path = Path(sctx_dir) / tex["external"]
            if sctx_path.exists():
                print(f"Decoding texture: {tex['external']}...")
                images.append(decode_sctx(str(sctx_path)))
            else:
                print(f"WARNING: Missing texture: {sctx_path}", file=sys.stderr)
                images.append(None)
        else:
            images.append(None)
    return images
