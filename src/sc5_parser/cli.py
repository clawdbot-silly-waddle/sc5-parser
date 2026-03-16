"""Command-line interface for sc5-parser."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sc5_parser.parser import SC5File
from sc5_parser.champion_card import render_champion_card
from sc5_parser.sctx import decode_sctx


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    # Route to subcommand if first arg is a known subcommand name
    _subcommands = {"render-card"}
    if argv and argv[0] in _subcommands:
        _dispatch_subcommand(argv)
        return

    # Legacy positional-file mode
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
        "--frame",
        type=int,
        metavar="N",
        help="Frame index (0-based) to render for the top-level MC",
    )
    ap.add_argument(
        "--all-frames",
        action="store_true",
        help="Export every frame of the top-level MC as separate images",
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


def _dispatch_subcommand(argv: list[str]) -> None:
    """Parse and dispatch subcommand-based invocations."""
    ap = argparse.ArgumentParser(prog="sc5-parser")
    sub = ap.add_subparsers(dest="command")

    card_p = sub.add_parser(
        "render-card",
        help="Render a champion card composite with portrait and overlay.",
    )
    card_p.add_argument(
        "--card-sc", required=True,
        help="Path to the card UI .sc file (e.g. ui_card_items.sc)",
    )
    card_p.add_argument(
        "--card-sctx",
        help="Path to the card .sctx file "
        "(default: auto-detect from card-sc basename)",
    )
    card_p.add_argument(
        "--portrait-sc", required=True,
        help="Path to the portrait .sc file (e.g. ui_card_knight_hero.sc)",
    )
    card_p.add_argument(
        "--portrait-sctx",
        help="Path to the portrait .sctx file "
        "(default: auto-detect from portrait-sc basename)",
    )
    card_p.add_argument(
        "--forms", required=True,
        help="Comma-separated form labels. First = primary (notch/glow/"
        "right diamond), second = secondary (left diamond). "
        "E.g. 'hero_unlocked,evo_unlocked' or just 'hero_unlocked'.",
    )
    card_p.add_argument(
        "--portrait-scale", type=float, default=0.55,
        help="Scale factor for the portrait (default: 0.55)",
    )
    card_p.add_argument(
        "--zoom", type=int, default=1,
        help="Upscale factor using nearest-neighbor (default: 1)",
    )
    card_p.add_argument(
        "-o", "--output", default="card.png",
        help="Output PNG path (default: card.png)",
    )

    args = ap.parse_args(argv)
    if args.command == "render-card":
        _render_card(args)
    else:
        ap.print_help()
        sys.exit(1)


# ------------------------------------------------------------------
def _print_info(sc: SC5File) -> None:
    print(f"File:        {sc.sc_path}")
    print(f"Shapes:      {len(sc.shapes)}")
    print(f"Movie clips: {len(sc.movie_clip_data)}")
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
            f"regions={', '.join(regions) or '-'})"
        )


def _extract(sc: SC5File, args: argparse.Namespace) -> None:
    sctx_dir = args.sctx_dir or str(sc.sc_path.parent)
    output_dir = args.output_dir or os.path.join(
        str(sc.sc_path.parent), "extracted"
    )
    os.makedirs(output_dir, exist_ok=True)

    texture_images = _load_textures(sc, sctx_dir)

    child_labels = _parse_child_labels(args.child_labels)
    frame_index = args.frame
    all_frames = args.all_frames

    names = args.extract if args.extract else list(sc.exports.keys())

    extracted = 0
    for name in names:
        if all_frames:
            extracted += _extract_all_frames(
                sc, name, texture_images, output_dir,
                frame_label=args.frame_label,
                child_labels=child_labels,
            )
        else:
            out_path = os.path.join(output_dir, f"{name}.png")
            result = sc.extract_sprite(
                name, texture_images, out_path,
                frame_label=args.frame_label,
                child_labels=child_labels,
                frame_index=frame_index,
            )
            if result:
                print(f"  Extracted: {name} ({result.width}×{result.height})")
                extracted += 1
            else:
                print(f"  Skip: {name} (no visible shapes)")

    print(f"\nExtracted {extracted}/{len(names)} sprites to {output_dir}")


def _extract_all_frames(
    sc: SC5File,
    name: str,
    texture_images: list,
    output_dir: str,
    frame_label: str | None = None,
    child_labels: dict[int, str] | None = None,
) -> int:
    """Export every frame of *name* as separate images. Returns count."""
    info = sc.get_export_frame_info(name)
    if info is None:
        print(f"  Skip: {name} (not found or not a movie clip)")
        return 0
    count = info["frame_count"]
    labels = info["frame_labels"]
    extracted = 0
    for fi in range(count):
        label_suffix = f"_{labels[fi]}" if fi < len(labels) and labels[fi] else ""
        out_path = os.path.join(output_dir, f"{name}_frame{fi}{label_suffix}.png")
        result = sc.extract_sprite(
            name, texture_images, out_path,
            frame_label=frame_label,
            child_labels=child_labels,
            frame_index=fi,
        )
        if result:
            print(f"  Frame {fi}{label_suffix}: {name} ({result.width}×{result.height})")
            extracted += 1
        else:
            print(f"  Frame {fi}{label_suffix}: (empty)")
    return extracted


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


def _auto_sctx(sc_path: str, explicit: str | None) -> str:
    """Resolve an .sctx path: use *explicit* if given, else try ``<base>_0.sctx``."""
    if explicit:
        return explicit
    p = Path(sc_path)
    candidate = p.parent / (p.stem + "_0.sctx")
    if candidate.exists():
        return str(candidate)
    return str(p.parent / (p.stem + ".sctx"))


def _render_card(args: argparse.Namespace) -> None:
    """Handle the ``render-card`` subcommand."""

    forms = [f.strip() for f in args.forms.split(",")]
    primary = forms[0]
    secondary = forms[1] if len(forms) > 1 else None

    if not primary or (secondary is not None and not secondary):
        print("ERROR: Form labels cannot be empty.", file=sys.stderr)
        sys.exit(1)

    kwargs: dict[str, str] = {"primary_form": primary}
    if secondary is not None:
        kwargs["secondary_form"] = secondary

    card_sctx = _auto_sctx(args.card_sc, args.card_sctx)
    portrait_sctx = _auto_sctx(args.portrait_sc, args.portrait_sctx)

    print(f"Loading card SC:       {args.card_sc}")
    print(f"Loading card texture:  {card_sctx}")
    card_sc = SC5File(args.card_sc)
    card_tex = _load_textures(card_sc, str(Path(card_sctx).parent))

    print(f"Loading portrait SC:   {args.portrait_sc}")
    print(f"Loading portrait tex:  {portrait_sctx}")
    portrait_sc = SC5File(args.portrait_sc)
    portrait_tex = _load_textures(portrait_sc, str(Path(portrait_sctx).parent))

    print(f"Forms: primary={primary}, secondary={secondary or '(default)'}")
    print(f"Portrait scale: {args.portrait_scale}")

    result = render_champion_card(
        card_sc, card_tex,
        portrait_sc, portrait_tex,
        portrait_scale=args.portrait_scale,
        render_scale=float(args.zoom),
        **kwargs,
    )

    if result is None:
        print("ERROR: Card rendering failed.", file=sys.stderr)
        sys.exit(1)

    result.image.save(args.output)
    print(f"Saved: {args.output} ({result.image.width}×{result.image.height})")
