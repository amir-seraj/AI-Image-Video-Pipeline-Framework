"""Shared image utilities for Casadei workflows."""
from __future__ import annotations

import math
from pathlib import Path

import yaml
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Camera preset loading
# ---------------------------------------------------------------------------

_PRESETS_PATH = Path(__file__).parent / "camera_presets.yaml"
_cached_presets: dict | None = None


def load_camera_presets() -> dict:
    """Load and cache camera presets from YAML."""
    global _cached_presets
    if _cached_presets is None:
        with open(_PRESETS_PATH) as f:
            _cached_presets = yaml.safe_load(f)
    return _cached_presets


def get_camera_preset(angle: str, foot: str = "pair") -> dict[str, str]:
    """Resolve a camera preset by angle name and foot variant.

    Returns dict with 'camera_desc' and 'staging_desc' keys.
    Falls back to using the angle string as-is if not found.
    """
    data = load_camera_presets()
    key = angle.lower().strip()
    # Resolve aliases
    aliases = data.get("aliases", {})
    if key in aliases:
        key = aliases[key]
    presets = data.get("presets", {})
    if key in presets:
        return presets[key][foot]
    return {
        "camera_desc": angle,
        "staging_desc": "The shoe(s) placed on the white surface.",
    }


def get_judge_notes() -> str:
    """Return the camera angle judge evaluation rubric."""
    return load_camera_presets()["judge_notes"]


def get_canonical_angles() -> list[str]:
    """Return the ordered list of canonical angle names."""
    return load_camera_presets()["canonical_angles"]


def get_pair_angles() -> set[str]:
    """Return angles that default to showing a pair."""
    return set(load_camera_presets()["pair_angles"])


def get_single_angles() -> set[str]:
    """Return angles that default to showing a single shoe."""
    return set(load_camera_presets()["single_angles"])


def foot_for_angle(angle: str, foot: str, single: bool = False) -> str:
    """Return the effective foot variant for a given angle.

    Pair angles use 'pair' unless single=True.
    Single angles use the provided foot value.
    """
    if single:
        return foot
    key = angle.lower().strip()
    if key in get_pair_angles():
        return "pair"
    return foot


# ---------------------------------------------------------------------------
# Foot framing prompt fragments
# ---------------------------------------------------------------------------

def foot_framing(foot: str, emphatic: bool = True) -> str:
    """Return a foot-specific prompt fragment.

    emphatic=True (default): detailed instructions for angle generation
    where reference images may show pairs.
    emphatic=False: simpler instructions for sketch-to-shoe generation.
    """
    if emphatic:
        if foot == "pair":
            return (
                "IMPORTANT — NUMBER OF SHOES: The output MUST contain exactly TWO shoes "
                "(a matching pair — left and right) placed side by side. "
                "The reference image shows a pair; keep both shoes in the output. "
                "Do NOT show only one shoe."
            )
        elif foot == "left":
            return (
                "IMPORTANT — NUMBER OF SHOES: The output MUST contain exactly ONE shoe — "
                "the LEFT shoe only, centered in the frame. "
                "Even though the reference image may show two shoes, generate ONLY the "
                "left shoe. Do NOT include the right shoe. Only one shoe in the image."
            )
        else:
            return (
                "IMPORTANT — NUMBER OF SHOES: The output MUST contain exactly ONE shoe — "
                "the RIGHT shoe only, centered in the frame. "
                "Even though the reference image may show two shoes, generate ONLY the "
                "right shoe. Do NOT include the left shoe. Only one shoe in the image."
            )
    else:
        if foot == "pair":
            return (
                "Show a matching pair of shoes — both left and right — "
                "centered side by side."
            )
        elif foot == "left":
            return "Show the left shoe only, centered and fully visible."
        else:
            return "Show the right shoe only, centered and fully visible."


# ---------------------------------------------------------------------------
# Aspect ratio helpers
# ---------------------------------------------------------------------------

MAX_INPUT_SIZE = 1024

SUPPORTED_RATIOS: list[tuple[int, int]] = [
    (1, 1), (1, 4), (1, 8),
    (2, 3), (3, 2), (3, 4), (4, 3),
    (4, 5), (5, 4),
    (8, 1), (9, 16), (16, 9), (21, 9),
]


def find_ratio(w: int, h: int) -> tuple[int, int]:
    """Find the nearest supported aspect ratio for dimensions w x h."""
    target = w / h
    return min(SUPPORTED_RATIOS, key=lambda r: abs(r[0] / r[1] - target))


def pad_to_ratio(
    img: PILImage.Image,
    ratio: tuple[int, int],
    max_size: int = MAX_INPUT_SIZE,
) -> PILImage.Image:
    """Pad and scale an image to the given aspect ratio, fitting within max_size."""
    wr, hr = ratio
    orig_w, orig_h = img.size

    if orig_w / orig_h <= wr / hr:
        canvas_h = orig_h
        canvas_w = round(orig_h * wr / hr)
    else:
        canvas_w = orig_w
        canvas_h = round(orig_w * hr / wr)

    scale = max_size / max(canvas_w, canvas_h)
    final_w = round(canvas_w * scale)
    final_h = round(canvas_h * scale)

    img_w = round(orig_w * scale)
    img_h = round(orig_h * scale)
    scaled = img.resize((img_w, img_h), PILImage.LANCZOS)

    canvas = PILImage.new("RGB", (final_w, final_h), (255, 255, 255))
    canvas.paste(scaled, ((final_w - img_w) // 2, (final_h - img_h) // 2))
    return canvas


# ---------------------------------------------------------------------------
# Sketch grid assembly
# ---------------------------------------------------------------------------

def build_sketch_grid(
    images: list[PILImage.Image],
    spacing: int = 20,
) -> PILImage.Image:
    """Assemble multiple sketch images into a square grid."""
    if not images:
        raise ValueError("No sketch images provided.")

    n = len(images)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)

    total_w = cols * cell_w + (cols + 1) * spacing
    total_h = rows * cell_h + (rows + 1) * spacing
    grid = PILImage.new("RGB", (total_w, total_h), (255, 255, 255))

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = spacing + col * (cell_w + spacing) + (cell_w - img.width) // 2
        y = spacing + row * (cell_h + spacing) + (cell_h - img.height) // 2
        grid.paste(img, (x, y))

    gw, gh = grid.size
    if gw != gh:
        size = max(gw, gh)
        square = PILImage.new("RGB", (size, size), (255, 255, 255))
        square.paste(grid, ((size - gw) // 2, (size - gh) // 2))
        return square

    return grid


# ---------------------------------------------------------------------------
# Material grid assembly
# ---------------------------------------------------------------------------

_TILE_SIZE = 500
_LABEL_HEIGHT = 30
_LABEL_FONT_SIZE = 28
_TILE_SPACING = 20
_MAX_GRID_PX = 2048


def _resolve_material_names(materials: list[dict]) -> list[str]:
    """Assign default names to materials that have name=None.

    Materials get "Material N", colors get "Color N".
    Numbering is per-type, independent.
    """
    mat_counter = 0
    color_counter = 0
    names: list[str] = []
    for entry in materials:
        if entry.get("name"):
            names.append(entry["name"])
        elif entry.get("is_color"):
            color_counter += 1
            names.append(f"Color {color_counter}")
        else:
            mat_counter += 1
            names.append(f"Material {mat_counter}")
    return names


def _build_labeled_tile(
    img: PILImage.Image,
    label: str,
    tile_w: int = _TILE_SIZE,
    tile_h: int = _TILE_SIZE,
    label_h: int = _LABEL_HEIGHT,
    font_size: int = _LABEL_FONT_SIZE,
) -> PILImage.Image:
    """Build a single tile: label strip on top, image fitted below."""
    from PIL import ImageDraw, ImageFont

    total_h = label_h + tile_h
    tile = PILImage.new("RGB", (tile_w, total_h), (255, 255, 255))

    # Draw label
    draw = ImageDraw.Draw(tile)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_x = (tile_w - text_w) // 2
    draw.text((text_x, 2), label, fill=(0, 0, 0), font=font)

    # Fit image into tile_w x tile_h box preserving aspect ratio
    text_gap = 8
    ow, oh = img.size
    scale = min(tile_w / ow, tile_h / oh)
    new_w = round(ow * scale)
    new_h = round(oh * scale)
    resized = img.convert("RGB").resize((new_w, new_h), PILImage.LANCZOS)
    paste_x = (tile_w - new_w) // 2
    paste_y = label_h + text_gap + (tile_h - new_h) // 2
    tile.paste(resized, (paste_x, paste_y))

    return tile


def build_material_grid(
    materials: list[dict],
) -> tuple[PILImage.Image, list[str]]:
    """Assemble material/color images into a labeled square grid.

    Each tile: label strip (~30px) above a 500x500 image box.
    Grid is padded to square. Downscaled if > 2048px on any side.

    Args:
        materials: List of dicts with 'image' (PIL.Image), 'name' (str|None),
                   'is_color' (bool). Other fields ignored here.

    Returns:
        (grid_image, resolved_names) — square PIL image and list of assigned names.
    """
    if not materials:
        raise ValueError("No materials provided.")

    names = _resolve_material_names(materials)

    tiles = []
    for entry, name in zip(materials, names):
        tile = _build_labeled_tile(entry["image"], name)
        tiles.append(tile)

    n = len(tiles)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    tile_w = tiles[0].width
    tile_h = tiles[0].height

    sp = _TILE_SPACING
    grid_w = cols * tile_w + (cols + 1) * sp
    grid_h = rows * tile_h + (rows + 1) * sp
    grid = PILImage.new("RGB", (grid_w, grid_h), (255, 255, 255))

    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        x = sp + col * (tile_w + sp)
        y = sp + row * (tile_h + sp)
        grid.paste(tile, (x, y))

    # Pad to square
    gw, gh = grid.size
    if gw != gh:
        size = max(gw, gh)
        square = PILImage.new("RGB", (size, size), (255, 255, 255))
        square.paste(grid, ((size - gw) // 2, (size - gh) // 2))
        grid = square

    # Downscale if exceeds max resolution
    w, h = grid.size
    if max(w, h) > _MAX_GRID_PX:
        scale = _MAX_GRID_PX / max(w, h)
        grid = grid.resize(
            (round(w * scale), round(h * scale)), PILImage.LANCZOS
        )

    return grid, names
